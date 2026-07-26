#!/usr/bin/env python3
"""
Compute technical indicators for the active stocks listed in config/symbols.csv
==============================================================================
Driven the same way as bulk_load.py:

    BASE/config/symbols.csv        -> list of symbols (uses active == 1).
                                      Optional 'kind' column (equity|index):
                                      kind=index rows are benchmarks for RS.
    BASE/raw/equity/SYMBOL.csv     -> raw OHLCV input  (Date, Open, High, Low, Close, Volume)
    BASE/ta/equity/SYMBOL_TA.csv   -> output with indicators appended per date

Indicators added (one value per date):
    RSI_14 (+ adaptive levels), MACD/PPO (+ normalized fields),
    EMA_9, EMA_21, EMA_50, EMA_200 (+ location/structure/slope/composite scores),
    BB_upper, BB_mid, BB_lower,
    ADX_14, ATR_14, Stoch_K, Stoch_D, VWAP,
    Trend_score / Trend_band / Trend_description
        (composite of EMA + Supertrend + ADX + market structure, weights in config),
    Momentum_score / Momentum_band / Momentum_description
        (composite of RSI + ROC + MACD + Stochastic, weights in config),
    Volatility_score / Volatility_band / Volatility_description
        (composite of BB + ATR + Donchian + Keltner; DIRECTIONLESS:
         0 = compressed, 100 = expanded, NOT bearish -> bullish),
    RS_* (relative strength vs benchmark indices; needs kind=index rows),
    DC_* (Donchian channel + breakout state),
    ROC_* (Rate of Change, multi-period + z-score + score),
    KC_* (Keltner channel + BB/KC squeeze)

USAGE
-----
    python compute_ta.py
    python compute_ta.py --base "C:/Data/Trading/StockData/market_data"
    python compute_ta.py --all          # ignore the active flag, process every listed symbol
"""

import argparse
from pathlib import Path
import sys
import traceback
import warnings
import numpy as np
import pandas as pd
import pandas_ta as ta

# ---- default base path mirrors bulk_load.py --------------------------------
# Resolved relative to this file (script/../ = market_data/) rather than
# hardcoded, so the same code works unmodified in any clone location --
# including a CI runner, where C:\Data\... simply doesn't exist.
DEFAULT_BASE = Path(__file__).resolve().parent.parent

# ---- DEFAULTS --------------------------------------------------------------
# Every value below can be overridden in  BASE/config/config.csv  (parameter,value).
# If that file is missing, or a parameter is absent from it, these defaults apply.
DEFAULTS = {
    # general
    "min_rows":            250,     # skip a symbol with fewer rows than this.
                                    # 250 (~1 trading year) is the point at which
                                    # EMA_200 and the 252-bar percentile/z-score
                                    # fields start producing real values. Lower it
                                    # to include newly-listed names, but expect
                                    # EMA_score_basis == "partial" on those rows.
    # adaptive RSI
    "rsi_length":          14,
    "rsi_hist_win":        252,     # rolling window for per-stock percentiles/z-score
    "rsi_upper_q":         0.90,    # per-stock overbought percentile
    "rsi_lower_q":         0.10,    # per-stock oversold percentile
    "rsi_full":            252,     # rows before trusting adaptive levels (else 70/30)
    "rsi_fixed_upper":     70,      # fallback overbought when history is thin
    "rsi_fixed_lower":     30,      # fallback oversold when history is thin
    # MACD / PPO
    "macd_fast":           12,
    "macd_slow":           26,
    "macd_signal":         9,
    "macd_hist_z_win":     252,     # lookback for MACD_hist z-score
    # other indicator periods
    "bb_length":           20,
    "bb_std":              2.0,
    "adx_length":          14,
    "atr_length":          14,
    "stoch_k":             14,
    "stoch_d":             3,
    # EMA ribbon periods (fast -> slow)
    "ema_periods":         "9,21,50,200",
    # EMA location scoring
    "loc_saturate_pct":    8.0,     # % distance treated as fully extended
    "loc_weights":         "0.35,0.30,0.22,0.13",   # per ema_periods, fast->slow
    # EMA slope scoring
    "slope_lookback":      "5,10,21,42",            # per ema_periods, fast->slow
    "slope_weights":       "0.30,0.28,0.24,0.18",   # per ema_periods, fast->slow
    "slope_saturate_mult": 0.12,    # saturation = lookback * this (scaled per EMA)
    # Supertrend
    "supertrend_length":   10,      # ATR period for the bands
    "supertrend_mult":     3.0,     # ATR multiplier (higher = fewer, slower flips)
    # Market structure (swing / pivot based)
    "ms_swing_n":          5,       # bars either side required to confirm a pivot
    "ms_range_tol_pct":    0.5,     # % tolerance when comparing swing levels
    # ---- volume analysis ----
    "vol_sma_fast":        20,      # short volume average
    "vol_sma_slow":        50,      # long volume average
    "vol_z_win":           252,     # lookback for volume z-score
    "vol_spike_mult":      2.0,     # Vol_ratio above this => "Yes" spike
    "vol_dry_mult":        0.5,     # Vol_ratio below this => dry-up
    "vwap_roll_win":       20,      # ROLLING VWAP window (daily-bar friendly)
    "obv_slope_lb":        20,      # lookback for OBV slope (% change)
    "cmf_length":          20,      # Chaikin Money Flow
    "mfi_length":          14,      # Money Flow Index ("volume RSI")
    "efi_length":          13,      # Elder Force Index
    "eom_length":          14,      # Ease of Movement
    "adosc_fast":          3,       # Chaikin Oscillator fast
    "adosc_slow":          10,      # Chaikin Oscillator slow
    "vwma_length":         20,      # Volume Weighted Moving Average
    "mfi_upper":           80,      # MFI overbought
    "mfi_lower":           20,      # MFI oversold
    # ---- Donchian Channel (breakout / range envelope) ----
    # NOTE: the window always EXCLUDES the current bar (shift 1). Including it
    # makes DC_upper == today's high on any new-high bar, so Close > DC_upper
    # could never be true and breakouts would never fire. Not configurable.
    "donchian_upper_len":  20,      # lookback for the highest high
    "donchian_lower_len":  20,      # lookback for the lowest low
    "donchian_prox_tol":   1.0,     # % from a band to be flagged "Near"
    # ---- Rate of Change (momentum, % over N bars) ----
    "roc_periods":         "5,10,21,63,126", # ~1w, 2w, 1m, 3m, 6m on daily bars
    "roc_z_win":           252,     # rolling window for ROC z-score
    "roc_z_full":          252,     # rows before ROC_z is labelled 'full'
    "roc_score_period":    63,      # which ROC feeds ROC_score / ROC_state
    "roc_saturate_pct":    25.0,    # % move treated as fully extended
    # ---- Keltner Channel (ATR envelope around an EMA) ----
    "keltner_length":      20,      # EMA period for the mid-line
    "keltner_atr_length":  10,      # ATR period for the band width
    "keltner_mult":        2.0,     # ATR multiplier
    "squeeze_bb_len":      20,      # BB length used for the BB/KC squeeze test
    "squeeze_bb_std":      2.0,     # BB std used for the BB/KC squeeze test
    # Squeeze threshold. "ratio" = classic Bollinger-inside-Keltner (ratio < 1),
    # which with these defaults fires on a MAJORITY of bars and is not selective.
    # "pctl" instead flags the tightest N% of this stock's OWN squeeze-ratio
    # history -> genuinely unusual compression, and cross-stock comparable.
    "squeeze_mode":        "pctl",  # "pctl" (recommended) or "ratio"
    "squeeze_ratio_thr":   1.0,     # used when squeeze_mode = "ratio"
    "squeeze_pctl":        20,      # used when squeeze_mode = "pctl"
    "squeeze_pctl_win":    252,     # rolling window for the percentile
    # ---- relative strength vs benchmark indices ----
    # Which files are RS benchmarks is decided ENTIRELY by bench_symbols /
    # bench_labels below -- each entry in bench_symbols is the raw CSV filename
    # stem (e.g. NIFTY50 -> raw/equity/NIFTY50.csv). Field prefixes come from
    # bench_labels, so RS_NIFTY50_excess_pct / RS_NIFTY500_excess_pct etc.
    # Marking a symbols.csv row kind=index is a SEPARATE setting: it only tells
    # main() that symbol is a non-tradable index (no self-RS, no Final_decision)
    # -- it does not by itself make that symbol a benchmark for others' RS; for
    # that it must ALSO appear in bench_symbols.
    "bench_symbols":       "NIFTY50,NIFTY500",   # filename stems, in raw/equity
    "bench_labels":        "NIFTY50,NIFTY500",   # prefixes used in field names
    "rs_slope_lb":         21,      # bars for the RS-line slope (~1 month)
    "rs_pctl_win":         252,     # rolling window for RS percentile rank
    "rs_roc_period":       63,      # horizon for excess return / RS_score
    "rs_saturate_pct":     20.0,    # outperformance % treated as fully extended
    "rs_primary":          "NIFTY50",   # which benchmark drives RS_score/RS_state
    # ---- composite Trend Score (EMA + Supertrend + ADX + Market Structure) ----
    # Weights are renormalized over whatever components are available on a bar,
    # so a missing component shifts weight to the others rather than dragging
    # the score toward neutral. Set any weight to 0 to drop that component.
    "trend_w_ema":         0.35,    # EMA ribbon: location + stacking + slope
    "trend_w_supertrend":  0.25,    # Supertrend: direction + distance from line
    "trend_w_adx":         0.20,    # ADX: strength, SIGNED by DI direction
    "trend_w_structure":   0.20,    # Market structure: HH/HL vs LH/LL
    # Supertrend sub-score: how far past the line counts as fully extended.
    # Beyond this the sub-score saturates, so one gap cannot dominate. Median
    # |ST_dist_pct| on daily equities sits near 5%, so 8 would leave the
    # component permanently mid-scale; 6 lets a normal trend reach ~90.
    "trend_st_saturate_pct": 6.0,
    # ADX sub-score: ADX value treated as a fully developed trend. The textbook
    # "strong trend" level is 40, but daily ADX_14 on NSE equities is below 20
    # roughly two thirds of the time -- using 40/20 leaves this component
    # contributing nothing on most bars. 32/15 tracks the real distribution.
    "trend_adx_saturate":  32.0,
    # ADX below this contributes no directional conviction (chop), so the ADX
    # sub-score stays at 50 rather than voting for a direction.
    "trend_adx_floor":     15.0,
    # Readable bands for Trend_score (same cut points as the EMA bands)
    "trend_band_strong_bull": 75.0,
    "trend_band_bull":        60.0,
    "trend_band_neutral":     40.0,
    "trend_band_bear":        25.0,
    # Agreement: how many components must sit on the same side of 50 before the
    # score is called "confirmed" rather than "mixed".
    "trend_agree_min":     3,
    # Minimum components that must be present before Trend_score is emitted at
    # all. Without this the score appears from bar ~9 backed by the fastest EMA
    # alone, which looks authoritative but is nearly meaningless.
    "trend_min_components": 3,
    # ---- composite Momentum Score (RSI + ROC + MACD + Stochastic) ----------
    # Same renormalizing blend as the trend score: weights are shared out over
    # whatever components exist on a bar, so a missing one shifts weight to the
    # others rather than dragging the score toward neutral. Set any to 0 to drop.
    "mom_w_rsi":           0.30,    # RSI: normalized position, mean-reverting
    "mom_w_roc":           0.30,    # ROC: raw rate of change over the headline horizon
    "mom_w_macd":          0.25,    # MACD histogram: acceleration of the trend
    "mom_w_stoch":         0.15,    # Stochastic %K: position within recent range
    # MACD histogram is unbounded and scale-dependent, so it is z-scored against
    # its own history and squashed. This is the z-value treated as fully extended.
    "mom_macd_z_sat":      2.0,
    # Stochastic is the noisiest of the four; its raw %K is pulled toward 50 by
    # this factor so a single whipsaw cannot swing the composite.
    "mom_stoch_damp":      0.80,
    # Readable bands for Momentum_score
    "mom_band_strong_bull": 70.0,
    "mom_band_bull":        58.0,
    "mom_band_neutral":     42.0,
    "mom_band_bear":        30.0,
    # Agreement: components on the same side of 50 before momentum is "confirmed"
    "mom_agree_min":       3,
    # Components required before Momentum_score is emitted at all
    "mom_min_components":  3,
    # Overbought/oversold guard: when RSI and Stochastic are BOTH at an extreme
    # the move is stretched, which the description flags rather than hides.
    "mom_extreme_rsi_hi":  70.0,
    "mom_extreme_rsi_lo":  30.0,
    # ---- composite Volatility Score (BB + ATR + Donchian + Keltner) --------
    # IMPORTANT: this composite is NOT directional. Trend_score and
    # Momentum_score run bearish(0) -> bullish(100); Volatility_score runs
    # COMPRESSED(0) -> EXPANDED(100). 50 is average volatility for this stock,
    # not "neutral bias". Never rank stocks bullish/bearish on this field.
    #
    # All four components are percentile-based against the stock's OWN history,
    # which is what makes them comparable across a universe: a 3% daily range is
    # extreme for a large-cap bank and unremarkable for a smallcap.
    "vol_w_bb":            0.30,    # Bollinger bandwidth percentile
    "vol_w_atr":           0.30,    # ATR percentile (true range, gap-aware)
    "vol_w_donchian":      0.20,    # Donchian channel width percentile
    "vol_w_keltner":       0.20,    # Keltner channel width percentile
    # Donchian and Keltner widths are raw percentages, so they are ranked against
    # the stock's own history over this window to match BB_bandwidth_pctl/ATR_pctl.
    "vol_width_pctl_win":  252,
    # Readable bands for Volatility_score (percentile space, so these are
    # percentile cut points rather than the +/-50 bands used by trend/momentum)
    "vol_band_very_low":   20.0,
    "vol_band_low":        40.0,
    "vol_band_high":       60.0,
    "vol_band_very_high":  80.0,
    # Components required before Volatility_score is emitted at all
    "vol_min_components":  3,
    # Squeeze consensus: how many of the squeeze-capable components (BB, Keltner)
    # must flag compression before the description calls it a squeeze.
    "vol_squeeze_min":     2,
    # Expansion detection: rise in Volatility_score over this lookback that
    # counts as a genuine volatility breakout rather than drift.
    "vol_expansion_lb":    10,
    "vol_expansion_jump":  20.0,
    # Bars after a confirmed squeeze within which an expansion is still called a
    # "release". A squeeze and an expansion never coincide on the SAME bar --
    # the bands have widened by then -- so the release must be detected as
    # expansion shortly AFTER compression.
    "vol_release_window":  15,
    # ---- output column visibility -----------------------------------------
    # Fields listed in 'hidden_fields' are COMPUTED as normal but dropped from
    # the written CSV. Nothing downstream changes: every score, state and
    # description is still derived from the full set, so hiding is purely
    # cosmetic and fully reversible.
    #
    # To bring a field back, delete its name from hidden_fields in config.csv
    # (or set show_all_fields to 1 to override the whole list at once). Because
    # the value is a plain comma-separated string, config.csv can hold it on one
    # long row; whitespace and blank entries are ignored.
    #
    # Unknown names are reported at load time rather than silently ignored, so a
    # typo shows up instead of quietly leaving a column visible.
    "show_all_fields":     0,       # 1 = ignore hidden_fields, emit everything
    "hidden_fields": (
        # EMA ribbon internals -- EMA_score / EMA_trend carry the summary
        "EMA_9,EMA_21,px_vs_EMA_9,px_vs_EMA_21,px_vs_EMA_50,px_vs_EMA_200,"
        "EMA_9_slope,EMA_21_slope,EMA_50_slope,EMA_200_slope,"
        "EMA_location_score,EMA_structure_score,EMA_slope_score,"
        "EMA_ribbon_n,EMA_score_basis,"
        # Supertrend internals -- ST_state / ST_signal carry the summary
        "ST_line,ST_dir,ST_dist_pct,ST_bars_since_flip,"
        # Market structure internals -- MS_structure / MS_bias carry the summary
        "MS_event,MS_swing_high,MS_swing_low,MS_last_HH,MS_last_HL,"
        "MS_last_LH,MS_last_LL,"
        # Trend composite internals -- Trend_score / band / description remain
        "Trend_supertrend_sub,Trend_adx_sub,Trend_structure_sub,"
        "Trend_bull_components,Trend_bear_components,Trend_components_n,"
        "Trend_score_basis,"
        # Relative strength per-benchmark detail -- RS_score / state / agreement remain
        "RS_NIFTY50_excess_pct,RS_NIFTY50_slope,RS_NIFTY50_pctl,"
        "RS_NIFTY500_excess_pct,RS_NIFTY500_slope,RS_NIFTY500_pctl,"
        # Momentum internals -- RSI_14 / MACD / states / composite remain
        "RSI_upper,RSI_lower,RSI_z,RSI_basis,MACD_hist,PPO,MACD_hist_z,"
        "Stoch_K,Stoch_D,Stoch_cross,"
        # ROC_5 / ROC_10 (performance vs the previous 5 / 10 trading days) stay
        # VISIBLE -- they're the plain % change a reader actually wants; only
        # the longer horizons and the z-score internals (all periods) hide.
        "ROC_21,ROC_63,ROC_126,"
        "ROC_5_z,ROC_10_z,ROC_21_z,ROC_63_z,ROC_126_z,"
        "ROC_5_z_basis,ROC_10_z_basis,ROC_21_z_basis,ROC_63_z_basis,ROC_126_z_basis,"
        "Mom_macd_sub,Momentum_bull_components,Momentum_bear_components,"
        "Momentum_distinct_views,Momentum_score_basis,"
        # Volatility internals -- Volatility_score / band / description remain
        "BB_upper,BB_mid,BB_lower,BB_pctB,BB_bandwidth,BB_bandwidth_pctl,BB_squeeze,"
        "ADX_14,DI_plus,DI_minus,ATR_14,ATR_pct,ATR_pctl,ATR_regime,"
        "DC_upper,DC_mid,DC_lower,DC_width_pct,DC_pct_rank,DC_bars_since_breakout,"
        "KC_upper,KC_mid,KC_lower,KC_width_pct,KC_pct_rank,KC_state,"
        "KC_squeeze_ratio,KC_squeeze_pctl,KC_squeeze,KC_squeeze_bars,"
        "KC_squeeze_release,BB_width_pctl,DC_width_pctl,KC_width_pctl,"
        "Volatility_squeeze_n,Volatility_bars_since_squeeze,"
        "Volatility_distinct_views,Volatility_gap_premium,"
        "Volatility_components_n,Volatility_score_basis,"
        # Risk levels -- ATR_note carries the readable summary
        "Stop_long,Stop_short,Setup,"
        # Volume internals -- Vol_state / Vol_trend / CMF / MFI remain
        "VWAP,VWAP_roll,px_vs_VWAP_pct,px_vs_VWAP_roll_pct,VWMA,"
        "Vol_SMA_fast,Vol_SMA_slow,Vol_ratio,Vol_z,Vol_pctile,Vol_spike,"
        "OBV,OBV_slope,AD,ADOSC,PVT,EFI,EOM,"
        # Stage internals -- Stage_name / substage / confidence / days remain
        "Stage,Stage_maturity_pct,Stage_start_date,Stage_start_price,"
        "Stage_return_pct,Stage_transition,Prev_stage,Stage_MA,Stage_MA_slope"
    ),
    # ---- final decision layer ---------------------------------------------
    # Combines Stage + Trend + Momentum + Volume + Volatility into one readable
    # call. NOTE: this is a SEQUENTIAL GATE, not an average of the five scores.
    # Averaging them would be wrong twice over: Volatility_score is a magnitude
    # (compressed->expanded), not a direction, so adding it to a directional
    # score is a category error; and Trend/Momentum share inputs, so averaging
    # double-counts recent price change. The gate also preserves WHY a setup
    # failed, which an average destroys.
    #
    # Order: Stage gates participation -> Trend sets direction -> Momentum and
    # Volume can only VETO or downgrade -> Volatility sizes the position.
    "decide_trend_long":   60.0,    # Trend_score above this = long bias
    "decide_trend_short":  40.0,    # Trend_score below this = short bias
    "decide_stage_conf":   50.0,    # Stage_confidence below this = treat as unclear
    "decide_allow_short":  1,       # 0 = long-only book; Stage 4 becomes Avoid
    "decide_require_volume": 1,     # 1 = weak volume downgrades conviction
    "decide_vol_extended": 80.0,    # Volatility_score above this + expanding = late
    # Conviction score is reported alongside the decision so setups can be
    # ranked. It counts gates PASSED, it is not a blend of the five scores.
    "decide_conviction_max": 5,
    # ---- Final_score (0-100) ----------------------------------------------
    # A numeric companion to Final_decision, built so the two can never
    # disagree: the GATE sets a ceiling, and directional strength fills within
    # it. A stage-blocked or trend-ambiguous bar is capped low no matter how
    # strong other components look, so ranking on Final_score reproduces the
    # gate's priorities rather than fighting them. 50 = neutral (no edge either
    # way); >50 leans long-actionable, <50 leans short-actionable.
    # Ceilings per gate outcome (how far a bar can score once blocked there):
    "score_cap_stage_block":   50.0,   # stage says stand aside -> pinned at neutral
    "score_cap_trend_block":   62.0,   # direction unconfirmed / ambiguous
    "score_cap_momentum_veto": 70.0,   # stretched or diverging
    "score_cap_volume_veto":   80.0,   # thin participation
    "score_cap_vol_extended":  85.0,   # confirmed but late
    # Within the ceiling, how much each passing component contributes. These are
    # applied to the directional distance from 50, then re-centred.
    "score_w_trend":       0.45,   # the primary directional driver
    "score_w_momentum":    0.30,   # confirmation of that direction
    "score_w_volume":      0.15,   # participation quality
    "score_w_stage":       0.10,   # stage confidence / substage bonus
    # ---- stage analysis (Weinstein 4-stage) ----
    "stage_ma_length":     150,     # daily equivalent of the 30-week MA (30*5)
    "stage_ma_type":       "SMA",   # SMA or EMA
    "stage_slope_lb":      20,      # bars used to measure MA slope
    "stage_flat_pct":      0.50,    # |slope %| below this => MA counts as FLAT
                                    # (calibrated for WEEKLY bars; use ~0.05 if stage_weekly=0)
    "stage_min_days":      10,      # min bars before a stage change is confirmed
    "stage_confirm_score": 55,      # EMA_structure_score used to confirm the call
    "stage_early_pct":     33,      # < this % of typical duration => "early"
    "stage_late_pct":      100,     # > this % => "late" / extended
    "stage_weekly":        1,       # 1 = classify on resampled WEEKLY bars (Weinstein)
    "stage_typ_days_1":    40,      # fallback typical duration, stage 1 (basing)
    "stage_typ_days_2":    60,      # fallback typical duration, stage 2 (advancing)
    "stage_typ_days_3":    30,      # fallback typical duration, stage 3 (topping)
    "stage_typ_days_4":    45,      # fallback typical duration, stage 4 (declining)
    # ---- BB / ADX / Stochastic decision framework ----
    "bb_squeeze_win":      126,     # lookback for bandwidth percentile (~6 months)
    "bb_squeeze_pctl":     10,      # bandwidth below this percentile => squeeze
    "bb_pctb_low":         0.05,    # %B at/below this = at/through lower band
    "bb_pctb_high":        0.95,    # %B at/above this = at/through upper band
    "adx_no_trend":        20,      # ADX below this = no trend (range regime)
    "adx_trending":        25,      # ADX above this = trending regime
    "adx_strong":          40,      # ADX above this = very strong / late
    "adx_extreme":         50,      # ADX above this = exhaustion risk
    "adx_slope_lb":        5,       # bars used to test if ADX is rising
    "stoch_upper":         80,      # Stochastic overbought
    "stoch_lower":         20,      # Stochastic oversold
    "stoch_pullback_lo":   20,      # trend-pullback entry zone, lower
    "stoch_pullback_hi":   40,      # trend-pullback entry zone, upper
    # ---- ATR-based risk / volatility decision fields ----
    "atr_stop_mult":       2.0,     # stop distance = ATR * this (2.0 = common default)
    "atr_pctl_win":        252,     # lookback for ATR percentile / regime
    "atr_low_pctl":        25,      # ATR pctile below this = Low volatility
    "atr_high_pctl":       75,      # above this = High volatility
    "atr_extreme_pctl":    90,      # above this = Extreme volatility
    "atr_slope_lb":        10,      # bars used to test if ATR is rising
    "atr_wide_stop_pct":   8.0,     # stop wider than this % of price = oversized risk
    "stoch_cross_window":  3,       # bars within which a bullish cross still counts
                                    # (same-bar-only is far too strict in practice)
    # composite weights (structure / slope / location)
    "w_structure":         0.40,
    "w_slope":             0.35,
    "w_location":          0.25,
    # Components required (of location/structure/slope) before EMA_score is
    # emitted at all -- mirrors trend_min_components/mom_min_components/
    # vol_min_components, so a lone sub-score cannot pass for a fully-backed one.
    "ema_min_components":  2,
    # EMA_trend bands on the composite score
    "band_strong_bull":    80,
    "band_bull":           60,
    "band_neutral":        40,
    "band_bear":           20,
}


def load_config(cfg_path: Path):
    """Read BASE/config/config.csv (parameter,value) over the DEFAULTS.

    Missing file or missing keys simply fall back to DEFAULTS, so the script
    always runs. Unknown keys are reported and ignored."""
    cfg = dict(DEFAULTS)
    if not cfg_path.exists():
        print(f"  config: {cfg_path.name} not found -> using defaults")
        _validate_trend_weights(cfg)
        return cfg

    raw = pd.read_csv(cfg_path)
    cols = {c.lower().strip(): c for c in raw.columns}
    if "parameter" not in cols or "value" not in cols:
        print(f"  config: {cfg_path.name} needs 'parameter,value' columns -> using defaults")
        _validate_trend_weights(cfg)
        return cfg

    pcol, vcol = cols["parameter"], cols["value"]
    applied, unknown = 0, []
    for _, r in raw.iterrows():
        key = str(r[pcol]).strip().lower()
        val = r[vcol]
        if key in ("", "nan") or pd.isna(r[pcol]):
            continue
        # Allow '#' section headers and blank spacer rows so config.csv can be
        # organised and self-documenting without them being flagged as unknown.
        if key.startswith("#"):
            continue
        if key not in DEFAULTS:
            unknown.append(key)
            continue
        if pd.isna(val) or str(val).strip() == "":
            continue                      # blank value -> keep default
        default = DEFAULTS[key]
        try:
            if isinstance(default, str):          # list-style params stay strings
                cfg[key] = str(val).strip()
            elif isinstance(default, int) and not isinstance(default, bool):
                cfg[key] = int(float(val))
            else:
                cfg[key] = float(val)
            applied += 1
        except (TypeError, ValueError):
            print(f"  config: bad value for '{key}' ({val!r}) -> keeping default")
    print(f"  config: {cfg_path.name} loaded ({applied} overrides)")
    if unknown:
        print(f"  config: ignored unknown parameter(s): {', '.join(unknown)}")
    _validate_trend_weights(cfg)
    return cfg


def _validate_trend_weights(cfg):
    """Sanity-check the composite weights and report what will be used.

    Every blend renormalizes over available components, so weights need not
    sum to 1.0 -- but a set that does not is usually a typo rather than an
    intent, and renormalization would hide it. A NEGATIVE weight is always
    wrong: it silently inverts that component, so a bullish RSI would push the
    score bearish. That is clamped rather than trusted."""
    groups = {
        "trend": {"ema": "trend_w_ema", "supertrend": "trend_w_supertrend",
                  "adx": "trend_w_adx", "structure": "trend_w_structure"},
        "momentum": {"rsi": "mom_w_rsi", "roc": "mom_w_roc",
                     "macd": "mom_w_macd", "stoch": "mom_w_stoch"},
        "volatility": {"bb": "vol_w_bb", "atr": "vol_w_atr",
                       "donchian": "vol_w_donchian", "keltner": "vol_w_keltner"},
        "ema_composite": {"location": "w_location", "structure": "w_structure",
                          "slope": "w_slope"},
        "final_score": {"trend": "score_w_trend", "momentum": "score_w_momentum",
                        "volume": "score_w_volume", "stage": "score_w_stage"},
    }
    for label, keys in groups.items():
        w = {}
        for name, k in keys.items():
            if k not in cfg:
                continue
            v = float(cfg[k])
            if v < 0:
                print(f"  config: {k}={v} is negative (would invert the "
                      f"component) -> using 0")
                cfg[k] = 0.0
                v = 0.0
            w[name] = v
        if not w:
            continue

        total = sum(w.values())
        if total <= 0:
            print(f"  config: all {label} weights are 0 -> "
                  f"{label.capitalize()}_score will be blank")
            continue

        if abs(total - 1.0) > 1e-6:
            shares = ", ".join(f"{nm} {v/total:.0%}" for nm, v in w.items() if v > 0)
            print(f"  config: {label} weights sum to {total:g}, not 1.0 -> "
                  f"effective split is {shares}")

        dropped = [nm for nm, v in w.items() if v == 0]
        if dropped:
            print(f"  config: {label} component(s) disabled (weight 0): "
                  f"{', '.join(dropped)}")

    # min_rows below rsi_full/roc_z_full means a symbol can pass the row-count
    # gate but never reach "adaptive"/"full" basis on those fields. Not wrong,
    # but worth flagging since it is easy to change one without the other.
    _min_rows = int(cfg.get("min_rows", 0))
    for _basis_key, _label in (("rsi_full", "RSI adaptive levels"),
                               ("roc_z_full", "ROC_*_z basis")):
        _full = int(cfg.get(_basis_key, 0))
        if _min_rows < _full:
            print(f"  config: min_rows ({_min_rows}) < {_basis_key} ({_full}) -> "
                  f"symbols just above min_rows will never reach 'full'/'adaptive' "
                  f"basis on {_label}")


def _num_list(text, cast=float):
    """Parse a comma-separated config value like '9,21,50,200'."""
    return [cast(x.strip()) for x in str(text).split(",") if str(x).strip() != ""]


def resolve_ema_config(cfg):
    """Turn the ribbon config strings into aligned dicts keyed by EMA period.

    Weight/lookback lists must match the number of EMA periods; if they don't,
    we fall back to sensible derived values rather than failing."""
    periods = _num_list(cfg["ema_periods"], int)

    def aligned(key, cast, fallback):
        vals = _num_list(cfg[key], cast)
        if len(vals) != len(periods):
            print(f"  config: '{key}' has {len(vals)} values but ema_periods has "
                  f"{len(periods)} -> using derived defaults for {key}")
            return fallback()
        return dict(zip(periods, vals))

    loc_w = aligned("loc_weights", float,
                    lambda: {p: 1.0 / len(periods) for p in periods})
    slope_lb = aligned("slope_lookback", int,
                       lambda: {p: max(3, int(p * 0.55)) for p in periods})
    slope_w = aligned("slope_weights", float,
                      lambda: {p: 1.0 / len(periods) for p in periods})
    return periods, loc_w, slope_lb, slope_w


def resolve_bench_config(cfg):
    """Pair benchmark file stems with the label used in their field names.

    Returns [(stem, label), ...]. Labels are sanitized so they are always valid
    column-name fragments. A mismatched label list falls back to the stems."""
    stems = [s.strip() for s in str(cfg["bench_symbols"]).split(",") if s.strip()]
    labels = [s.strip() for s in str(cfg["bench_labels"]).split(",") if s.strip()]
    if not stems:
        return []
    if len(labels) != len(stems):
        if labels:
            print(f"  config: 'bench_labels' has {len(labels)} values but "
                  f"'bench_symbols' has {len(stems)} -> using symbol names as labels")
        labels = list(stems)
    clean = []
    for st, lb in zip(stems, labels):
        lb = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in lb)
        clean.append((st, lb))
    return clean


def resolve_roc_config(cfg):
    """ROC periods from config, de-duplicated and sorted fast -> slow."""
    periods = sorted(set(_num_list(cfg["roc_periods"], int)))
    periods = [p for p in periods if p > 0]
    if not periods:
        print("  config: 'roc_periods' empty/invalid -> using 5,21,63,126")
        periods = [5, 21, 63, 126]
    return periods


def _as_series(result, index, name=""):
    """Normalize a pandas-ta return value into a float Series aligned to `index`.

    pandas-ta returns the INPUT DataFrame unchanged when the series is shorter
    than the indicator's period (e.g. ta.ema(length=200) on 90 rows), instead of
    a Series or an all-NaN column. Assigning that to a single column raises
    "Cannot set a DataFrame with multiple columns to the single column X".
    We convert any such non-Series result into an all-NaN Series so short-history
    symbols produce blank indicator columns rather than crashing the whole run."""
    if isinstance(result, pd.Series):
        return pd.to_numeric(result, errors="coerce")
    if isinstance(result, pd.DataFrame):
        # A genuine single-column indicator frame is usable; the passthrough of
        # the OHLCV input frame is not.
        if result.shape[1] == 1:
            return pd.to_numeric(result.iloc[:, 0], errors="coerce")
        if name:
            print(f"  warn: {name} unavailable (insufficient history) -> blank column")
        return pd.Series(np.nan, index=index, dtype=float)
    if result is None:
        if name:
            print(f"  warn: {name} returned nothing -> blank column")
        return pd.Series(np.nan, index=index, dtype=float)
    try:
        return pd.Series(np.asarray(result, dtype=float), index=index)
    except Exception:
        return pd.Series(np.nan, index=index, dtype=float)


def _pick(frame, prefix):
    """Return first column whose name starts with prefix (version-robust).

    Returns an all-NaN Series when `frame` is not a DataFrame or has no matching
    column -- which is what pandas-ta hands back on insufficient history."""
    if isinstance(frame, pd.DataFrame):
        for c in frame.columns:
            if c.startswith(prefix):
                return pd.to_numeric(frame[c], errors="coerce")
        return pd.Series(np.nan, index=frame.index, dtype=float)
    if isinstance(frame, pd.Series):
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.Series(dtype=float)


def _weighted_blend(n, components, weights, min_components):
    """Weighted average of named 0-100 component arrays, renormalized over
    whichever are non-NaN on a given row (a missing component shifts weight to
    the rest rather than dragging the result toward neutral).

    `components` is {name: np.ndarray}; `weights` is {name: float} (weights
    <= 0 drop that component entirely). Returns (blended, navail, basis):
    navail is the per-row count of components actually available, and basis is
    "full" / "partial" / "" (blank once the row is suppressed). The result is
    NaN on rows backed by fewer than min(min_components, positively-weighted
    component count) -- an early reading built on one fast component is not
    comparable to a fully-backed one, so it is withheld rather than shown
    looking equally confident. Shared by the Trend/Momentum/Volatility/EMA
    composite scores, which all use this exact pattern."""
    total = np.zeros(n); wsum = np.zeros(n); navail = np.zeros(n)
    n_weighted = 0
    for name, arr in components.items():
        w = weights.get(name, 0.0)
        if w <= 0:
            continue
        n_weighted += 1
        valid = ~np.isnan(arr)
        total = np.where(valid, total + np.nan_to_num(arr) * w, total)
        wsum  = np.where(valid, wsum + w, wsum)
        navail = np.where(valid, navail + 1, navail)
    with np.errstate(invalid="ignore", divide="ignore"):
        blended = np.where(wsum > 0, total / wsum, np.nan)
    blended = np.where(navail >= min(min_components, n_weighted), blended, np.nan)
    basis = np.where(navail >= n_weighted, "full", np.where(navail > 0, "partial", ""))
    basis = np.where(np.isnan(blended), "", basis)
    return blended, navail.astype(int), basis


def _bars_since(event):
    """Bars since `event` (a bool array) was last True (0 on the event bar
    itself); NaN before the first occurrence rather than a misleading
    synthetic zero. Shared by ST_bars_since_flip / DC_bars_since_breakout /
    Volatility_bars_since_squeeze, which all walk this identical pattern."""
    result = np.full(len(event), np.nan)
    last = np.nan
    for i in range(len(event)):
        if event[i]:
            last = i
        if not np.isnan(last):
            result[i] = i - last
    return result


def add_indicators(df, cfg=None, benchmarks=None, is_index=False):
    """df has columns Date, Open, High, Low, Close, Volume (Date already datetime).
    cfg is the dict from load_config(); omit it to use DEFAULTS.
    benchmarks is an optional {label: DataFrame} of index OHLCV used for the
    relative-strength block; omit it and the RS fields are simply absent.
    is_index marks the row as a benchmark index: Yahoo reports Volume as 0 for
    Indian indices, which would make every volume-derived field silently wrong,
    so those columns are blanked rather than filled with meaningless numbers.
    Returns df with indicator columns appended, one value per date."""
    if cfg is None:
        cfg = dict(DEFAULTS)
    EMA_PERIODS, LOC_WEIGHTS, SLOPE_LOOKBACK, SLOPE_WEIGHTS = resolve_ema_config(cfg)
    RSI_HIST_WIN = int(cfg["rsi_hist_win"])
    RSI_FULL     = int(cfg["rsi_full"])
    LOC_SATURATE_PCT = float(cfg["loc_saturate_pct"])
    COMPOSITE_W = {"structure": float(cfg["w_structure"]),
                   "slope":     float(cfg["w_slope"]),
                   "location":  float(cfg["w_location"])}
    # pandas-ta reads lowercase ohlcv by default; make a working copy
    work = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume"
    }).copy()

    out = df.copy()
    n = len(out)          # row count, needed by several blocks below

    # --- enforce proper warm-up: blank values that don't yet have a full lookback ---
    # pandas-ta's EWM-based indicators (RSI, MACD, EMA, ADX, ATR, Stoch) emit a
    # number from the very first rows, but a value computed on fewer than its full
    # period is NOT a valid reading. We mask those warm-up rows to NaN so the first
    # real RSI_14 appears at row 15 (14 price changes), etc. n = row position (0-based).
    # Defined here (rather than further down, where most masking happens) so
    # indicators that later blocks immediately READ -- ADX, ATR, Stochastic --
    # can be masked right after they're computed, before anything downstream
    # consumes their still-immature early values.
    idx = np.arange(n)

    def mask(col, warmup):
        """Set the first `warmup` rows (0-based positions < warmup) to NaN."""
        if col in out.columns:
            out.loc[idx < warmup, col] = np.nan

    # --- momentum ---
    rsi_len = int(cfg["rsi_length"])
    out["RSI_14"] = _as_series(work.ta.rsi(length=rsi_len), out.index, f"RSI_{rsi_len}")

    m_fast, m_slow, m_sig = int(cfg["macd_fast"]), int(cfg["macd_slow"]), int(cfg["macd_signal"])
    macd = work.ta.macd(fast=m_fast, slow=m_slow, signal=m_sig)
    out["MACD"]        = _pick(macd, "MACD_")
    out["MACD_hist"]   = _pick(macd, "MACDh")
    out["MACD_signal"] = _pick(macd, "MACDs")

    # PPO = MACD expressed as % of the slow EMA -> cross-stock-comparable magnitude
    ppo = work.ta.ppo(fast=m_fast, slow=m_slow, signal=m_sig)
    out["PPO"] = _pick(ppo, "PPO_")

    # --- trend / moving averages ---
    # --- trend / EMA ribbon (periods come from config) ---
    for _p in EMA_PERIODS:
        out[f"EMA_{_p}"] = _as_series(work.ta.ema(length=_p), out.index, f"EMA_{_p}")

    # --- volatility bands ---
    bb_len, bb_std = int(cfg["bb_length"]), float(cfg["bb_std"])
    bb = work.ta.bbands(length=bb_len, std=bb_std)
    out["BB_upper"] = _pick(bb, "BBU")
    out["BB_mid"]   = _pick(bb, "BBM")
    out["BB_lower"] = _pick(bb, "BBL")

    # --- trend strength / volatility ---
    adx_len = int(cfg["adx_length"])
    adx = work.ta.adx(length=adx_len)
    out["ADX_14"] = _pick(adx, "ADX_")
    out["DI_plus"]  = _pick(adx, "DMP")     # +DI : gives ADX its DIRECTION
    out["DI_minus"] = _pick(adx, "DMN")     # -DI
    # ADX/DI need ~2*period for DI smoothing + ADX smoothing; masked immediately
    # so every later block (ADX_regime, Trend_score, ...) reads a mature value.
    mask("ADX_14", adx_len * 2 - 1)
    mask("DI_plus", adx_len * 2 - 1)
    mask("DI_minus", adx_len * 2 - 1)
    atr_len = int(cfg["atr_length"])
    out["ATR_14"] = _as_series(work.ta.atr(length=atr_len), out.index, f"ATR_{atr_len}")
    mask("ATR_14", atr_len)

    # --- stochastic ---
    sk, sd = int(cfg["stoch_k"]), int(cfg["stoch_d"])
    stoch = work.ta.stoch(k=sk, d=sd)

    # --- Supertrend (ATR-band trend follower) ---
    st_len  = int(cfg["supertrend_length"])
    st_mult = float(cfg["supertrend_mult"])
    try:
        stf = work.ta.supertrend(length=st_len, multiplier=st_mult)
        out["ST_line"] = _pick(stf, "SUPERT_")
        out["ST_dir"]  = _pick(stf, "SUPERTd")
    except Exception as e:
        print(f"  warn: Supertrend unavailable ({type(e).__name__}: {e})")
        out["ST_line"] = np.nan
        out["ST_dir"]  = np.nan
    out["Stoch_K"] = _pick(stoch, "STOCHk")
    out["Stoch_D"] = _pick(stoch, "STOCHd")
    # Masked immediately: %K needs k+d-1 bars, %D one more smoothing pass, so
    # downstream state/crossover/Setup logic never reads an immature value.
    mask("Stoch_K", sk + sd - 1)
    mask("Stoch_D", sk + sd * 2 - 2)

    # --- VWAP ---
    # WARNING: pandas-ta's vwap() anchors PER CALENDAR DAY. On daily (EOD) bars each
    # row is its own anchor group, so the result collapses to the typical price
    # (H+L+C)/3 and carries NO volume information at all. It is kept only for
    # backward compatibility. Use VWAP_roll (rolling, volume-weighted) instead.
    try:
        w = work.set_index(pd.to_datetime(work["Date"]))
        out["VWAP"] = w.ta.vwap().values
    except Exception as e:
        print(f"  warn: VWAP unavailable ({type(e).__name__}: {e})")
        out["VWAP"] = np.nan

    # ================= VOLUME ANALYSIS ======================================
    v_fast = int(cfg["vol_sma_fast"]); v_slow = int(cfg["vol_sma_slow"])
    vol = out["Volume"].astype(float)
    tp = (out["High"].astype(float) + out["Low"].astype(float) + out["Close"].astype(float)) / 3.0

    # ---- 1. Volume level & relative volume ----
    out["Vol_SMA_fast"] = vol.rolling(v_fast).mean()
    out["Vol_SMA_slow"] = vol.rolling(v_slow).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["Vol_ratio"] = vol / out["Vol_SMA_fast"]      # today vs typical -> comparable
    vz_win = min(int(cfg["vol_z_win"]), n)
    vroll = vol.rolling(vz_win, min_periods=max(20, vz_win // 4))
    out["Vol_z"] = (vol - vroll.mean()) / vroll.std()
    out["Vol_pctile"] = vroll.rank(pct=True) * 100.0     # 0-100 within own history

    # ---- 2. Rolling VWAP (daily-bar friendly) + distances ----
    rw = int(cfg["vwap_roll_win"])
    pv = (tp * vol).rolling(rw).sum()
    vv = vol.rolling(rw).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["VWAP_roll"] = pv / vv
        _vwap_safe = out["VWAP"].replace(0, np.nan)
        _vwap_roll_safe = out["VWAP_roll"].replace(0, np.nan)
        out["px_vs_VWAP_pct"] = (out["Close"] - out["VWAP"]) / _vwap_safe * 100.0
        out["px_vs_VWAP_roll_pct"] = (out["Close"] - out["VWAP_roll"]) / _vwap_roll_safe * 100.0

    # ---- 3. Volume-weighted moving average ----
    try:
        out["VWMA"] = _as_series(work.ta.vwma(length=int(cfg["vwma_length"])), out.index, "VWMA")
    except Exception as e:
        print(f"  warn: VWMA unavailable ({type(e).__name__}: {e})")
        out["VWMA"] = np.nan

    # ---- 4. Accumulation / distribution family ----
    try:
        out["OBV"] = _as_series(work.ta.obv(), out.index, "OBV")
    except Exception as e:
        print(f"  warn: OBV unavailable ({type(e).__name__}: {e})")
        out["OBV"] = np.nan
    try:
        out["AD"] = _as_series(work.ta.ad(), out.index, "AD")
    except Exception as e:
        print(f"  warn: AD unavailable ({type(e).__name__}: {e})")
        out["AD"] = np.nan
    try:
        out["ADOSC"] = _pick(pd.DataFrame(work.ta.adosc(
            fast=int(cfg["adosc_fast"]), slow=int(cfg["adosc_slow"]))), "ADOSC")
    except Exception as e:
        print(f"  warn: ADOSC unavailable ({type(e).__name__}: {e})")
        out["ADOSC"] = np.nan
    try:
        out["PVT"] = _as_series(work.ta.pvt(), out.index, "PVT")
    except Exception as e:
        print(f"  warn: PVT unavailable ({type(e).__name__}: {e})")
        out["PVT"] = np.nan

    # ---- 5. Money-flow oscillators ----
    try:
        out["CMF"] = _as_series(work.ta.cmf(length=int(cfg["cmf_length"])), out.index, "CMF")
    except Exception as e:
        print(f"  warn: CMF unavailable ({type(e).__name__}: {e})")
        out["CMF"] = np.nan
    try:
        out["MFI"] = _as_series(work.ta.mfi(length=int(cfg["mfi_length"])), out.index, "MFI")
    except Exception as e:
        print(f"  warn: MFI unavailable ({type(e).__name__}: {e})")
        out["MFI"] = np.nan
    try:
        out["EFI"] = _as_series(work.ta.efi(length=int(cfg["efi_length"])), out.index, "EFI")
    except Exception as e:
        print(f"  warn: EFI unavailable ({type(e).__name__}: {e})")
        out["EFI"] = np.nan
    try:
        eom = work.ta.eom(length=int(cfg["eom_length"]))
        out["EOM"] = (_as_series(eom, out.index, "EOM")
                      if not isinstance(eom, pd.DataFrame) or eom.shape[1] == 1
                      else _pick(eom, "EOM"))
    except Exception as e:
        print(f"  warn: EOM unavailable ({type(e).__name__}: {e})")
        out["EOM"] = np.nan

    # ================= DONCHIAN CHANNEL =====================================
    # Highest high / lowest low over the PRIOR N bars (current bar excluded via
    # shift) so a "breakout" means price exceeded the ESTABLISHED range, rather
    # than the range being defined by today's own extreme.
    dc_up_len = int(cfg["donchian_upper_len"])
    dc_lo_len = int(cfg["donchian_lower_len"])
    DC_SHIFT  = 1                       # fixed by design; see DEFAULTS note

    _dc_h = out["High"].shift(DC_SHIFT)
    _dc_l = out["Low"].shift(DC_SHIFT)
    out["DC_upper"] = _dc_h.rolling(dc_up_len, min_periods=dc_up_len).max()
    out["DC_lower"] = _dc_l.rolling(dc_lo_len, min_periods=dc_lo_len).min()
    out["DC_mid"]   = (out["DC_upper"] + out["DC_lower"]) / 2.0

    with np.errstate(invalid="ignore", divide="ignore"):
        # width as % of mid -> range compression/expansion, cross-stock comparable
        out["DC_width_pct"] = ((out["DC_upper"] - out["DC_lower"])
                               / out["DC_mid"].replace(0, np.nan) * 100.0)
        # position in channel: 0 = at lower band, 100 = at upper band.
        # CLIPPED because Donchian bands are hard boundaries -- any excursion
        # outside is a breakout, which DC_state already reports explicitly.
        _dc_rng = (out["DC_upper"] - out["DC_lower"]).replace(0, np.nan)
        out["DC_pct_rank"] = ((out["Close"] - out["DC_lower"])
                              / _dc_rng * 100.0).clip(0, 100)

    # ================= RATE OF CHANGE =======================================
    # % change over N bars. Already scale-free, so the raw value is directly
    # comparable across stocks; the z-score adds per-stock context (is THIS
    # move large for THIS stock?).
    ROC_PERIODS = resolve_roc_config(cfg)
    roc_z_win = int(cfg["roc_z_win"])
    for _p in ROC_PERIODS:
        _r = out["Close"].pct_change(_p) * 100.0
        out[f"ROC_{_p}"] = _r
        _w = min(roc_z_win, n)
        _roll = _r.rolling(_w, min_periods=max(30, _w // 4))
        out[f"ROC_{_p}_z"] = (_r - _roll.mean()) / _roll.std()

    # ================= KELTNER CHANNEL ======================================
    # EMA mid-line with ATR-scaled bands. Unlike Bollinger (std-based), the
    # width reflects TRUE range, so gaps widen it -- materially better on NSE
    # results/policy days where std-based bands understate realized risk.
    kc_len     = int(cfg["keltner_length"])
    kc_atr_len = int(cfg["keltner_atr_length"])
    kc_mult    = float(cfg["keltner_mult"])
    try:
        _kc_mid = _as_series(work.ta.ema(length=kc_len), out.index, "KC_mid EMA")
        _kc_atr = _as_series(work.ta.atr(length=kc_atr_len), out.index, "KC ATR")
        out["KC_mid"]   = _kc_mid
        out["KC_upper"] = _kc_mid + kc_mult * _kc_atr
        out["KC_lower"] = _kc_mid - kc_mult * _kc_atr
    except Exception:
        out["KC_mid"] = np.nan; out["KC_upper"] = np.nan; out["KC_lower"] = np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        out["KC_width_pct"] = ((out["KC_upper"] - out["KC_lower"])
                               / out["KC_mid"].replace(0, np.nan) * 100.0)
        # NOT clipped (unlike DC_pct_rank): Keltner bands are soft, so HOW FAR
        # outside price sits is itself the signal.
        _kc_rng = (out["KC_upper"] - out["KC_lower"]).replace(0, np.nan)
        out["KC_pct_rank"] = ((out["Close"] - out["KC_lower"]) / _kc_rng * 100.0)

    # RSI(14): needs 14 changes -> first valid at position 14 (row 15)
    mask("RSI_14", rsi_len)

    # --- adaptive per-stock RSI levels (computed AFTER warmup mask) ---
    # Fixed 70/30 misidentifies overbought/oversold because each stock oscillates
    # in its own range. We derive thresholds from the stock's OWN RSI distribution:
    #   RSI_upper/lower : rolling 90th/10th percentile of that stock's RSI
    #   RSI_z           : RSI z-scored vs its own rolling mean/std (cross-stock comparable)
    #   RSI_state       : Overbought / Oversold / Neutral from that stock's levels
    #   RSI_basis       : 'adaptive' once enough history exists, else 'fixed_70_30'
    win = min(RSI_HIST_WIN, n)
    rsi = out["RSI_14"]
    roll = rsi.rolling(win, min_periods=max(30, win // 4))
    out["RSI_upper"] = roll.quantile(float(cfg["rsi_upper_q"]))
    out["RSI_lower"] = roll.quantile(float(cfg["rsi_lower_q"]))
    rmean = roll.mean()
    rstd  = roll.std()
    out["RSI_z"] = (rsi - rmean) / rstd

    # basis: adaptive only where we have a real percentile AND enough total history
    enough = (idx >= RSI_FULL) & out["RSI_upper"].notna() & out["RSI_lower"].notna()
    out["RSI_basis"] = np.where(enough, "adaptive", "fixed_70_30")

    # effective thresholds: adaptive where trusted, else fixed 70/30
    up_eff = np.where(enough, out["RSI_upper"], float(cfg["rsi_fixed_upper"]))
    lo_eff = np.where(enough, out["RSI_lower"], float(cfg["rsi_fixed_lower"]))

    # state from the effective per-row thresholds
    state = np.full(n, "Neutral", dtype=object)
    state[rsi.to_numpy() >= up_eff] = "Overbought"
    state[rsi.to_numpy() <= lo_eff] = "Oversold"
    state[rsi.isna().to_numpy()] = ""          # blank during RSI warmup
    out["RSI_state"] = state
    out["RSI_basis"] = np.where(rsi.isna().to_numpy(), "", out["RSI_basis"])
    # SMA/EMA: need `length` bars
    # EMA ribbon: each needs `length` bars
    for _p in EMA_PERIODS:
        mask(f"EMA_{_p}", _p)

    out = out.copy()          # defragment before the next column batch
    # ================= EMA RIBBON SCORING (A: location, B: structure, C: slope)
    # Each sub-score is 0-100 and normalized (% based), so it is comparable
    # ACROSS stocks regardless of price level.
    close_np = out["Close"].to_numpy(dtype=float)
    emas = {p: out[f"EMA_{p}"].to_numpy(dtype=float) for p in EMA_PERIODS}

    # ---- A. LOCATION: where is price relative to each EMA? -----------------
    # Per EMA: pct distance (close-ema)/ema*100, mapped to 0-100 with 50 = at EMA.
    # Saturates at +/-LOC_SATURATE_PCT so a parabolic extension doesn't dominate.
    loc_total = np.zeros(n); loc_wsum = np.zeros(n)
    for p in EMA_PERIODS:
        e = emas[p]
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = (close_np - e) / e * 100.0
            out[f"px_vs_EMA_{p}"] = np.round(pct, 4)   # keep raw % as a usable field
            sub = 50.0 + 50.0 * np.clip(pct / LOC_SATURATE_PCT, -1.0, 1.0)
        valid = ~np.isnan(sub)
        w = LOC_WEIGHTS[p]
        loc_total = np.where(valid, loc_total + sub * w, loc_total)
        loc_wsum  = np.where(valid, loc_wsum + w, loc_wsum)
    with np.errstate(invalid="ignore", divide="ignore"):
        location = np.where(loc_wsum > 0, loc_total / loc_wsum, np.nan)

    # ---- Yes/No flag: is price below the slowest EMA in the ribbon? ---------
    # Driven off the slowest configured EMA (EMA_200 by default) so it follows
    # `ema_periods` if that is changed in config.csv. Blank during warm-up.
    _slow = max(EMA_PERIODS)
    _slow_ema = emas[_slow]
    _below = np.full(n, "", dtype=object)
    _valid = (~np.isnan(_slow_ema)) & (~np.isnan(close_np))
    _below[_valid & (close_np < _slow_ema)] = "Yes"
    _below[_valid & (close_np >= _slow_ema)] = "No"
    out[f"Below_EMA_{_slow}"] = _below

    # ---- B. STRUCTURE: are the EMAs stacked in order? -----------------------
    # Score ALL 6 pairwise orderings (9>21, 9>50, 9>200, 21>50, 21>200, 50>200).
    # Using only adjacent pairs would discard information about partial stacks.
    # 100 = perfect bull stack (9>21>50>200), 0 = perfect inversion.
    pairs = [(EMA_PERIODS[a], EMA_PERIODS[b])
             for a in range(len(EMA_PERIODS)) for b in range(a + 1, len(EMA_PERIODS))]
    st_hits = np.zeros(n); st_valid = np.zeros(n)
    for fast, slow in pairs:
        f, s = emas[fast], emas[slow]
        ok = (~np.isnan(f)) & (~np.isnan(s))
        st_hits  = np.where(ok & (f > s), st_hits + 1.0, st_hits)
        st_valid = np.where(ok, st_valid + 1.0, st_valid)
    with np.errstate(invalid="ignore", divide="ignore"):
        structure = np.where(st_valid > 0, st_hits / st_valid * 100.0, np.nan)

    # ---- C. SLOPE: is each EMA rising or falling? --------------------------
    # % change over a lookback scaled to that EMA's own timeframe, mapped 0-100.
    # Saturation threshold also scales: a 200-EMA moving 2% over 42 bars is a
    # strong trend, whereas a 9-EMA needs more to count as equally steep.
    slope_total = np.zeros(n); slope_wsum = np.zeros(n)
    for p in EMA_PERIODS:
        lb = SLOPE_LOOKBACK[p]
        e_ser = out[f"EMA_{p}"]
        pct_chg = (e_ser / e_ser.shift(lb) - 1.0) * 100.0
        out[f"EMA_{p}_slope"] = pct_chg.round(4)
        sat = max(1.0, lb * float(cfg["slope_saturate_mult"]))   # scaled per EMA
        sub = 50.0 + 50.0 * np.clip(pct_chg.to_numpy() / sat, -1.0, 1.0)
        valid = ~np.isnan(sub)
        w = SLOPE_WEIGHTS[p]
        slope_total = np.where(valid, slope_total + sub * w, slope_total)
        slope_wsum  = np.where(valid, slope_wsum + w, slope_wsum)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(slope_wsum > 0, slope_total / slope_wsum, np.nan)

    out["EMA_location_score"]  = np.round(location, 2)
    out["EMA_structure_score"] = np.round(structure, 2)
    out["EMA_slope_score"]     = np.round(slope, 2)

    # ---- COMPOSITE: weighted blend, renormalized over available sub-scores ---
    comp_total = np.zeros(n); comp_wsum = np.zeros(n); comp_navail = np.zeros(n)
    for name, arr in (("location", location), ("structure", structure), ("slope", slope)):
        w = COMPOSITE_W[name]
        if w <= 0:
            continue
        valid = ~np.isnan(arr)
        comp_total = np.where(valid, comp_total + np.nan_to_num(arr) * w, comp_total)
        comp_wsum  = np.where(valid, comp_wsum + w, comp_wsum)
        comp_navail = np.where(valid, comp_navail + 1, comp_navail)
    with np.errstate(invalid="ignore", divide="ignore"):
        composite = np.where(comp_wsum > 0, comp_total / comp_wsum, np.nan)
    # Suppress bars backed by too few sub-scores -- the same guard Trend_score /
    # Momentum_score / Volatility_score already apply. Without it a bar with
    # only `location` available (the fastest-settling sub-score) can print a
    # confident-looking EMA_score indistinguishable from a fully-backed one.
    _ema_min = int(cfg["ema_min_components"])
    _ema_weighted = sum(1 for _nm in COMPOSITE_W if COMPOSITE_W[_nm] > 0)
    composite = np.where(comp_navail >= min(_ema_min, _ema_weighted), composite, np.nan)
    out["EMA_score"] = np.round(composite, 2)

    # readable band for the composite
    band = np.full(n, "", dtype=object)
    ok = ~np.isnan(composite)
    b_sbull = float(cfg["band_strong_bull"]); b_bull = float(cfg["band_bull"])
    b_neut  = float(cfg["band_neutral"]);      b_bear = float(cfg["band_bear"])
    band[ok & (composite >= b_sbull)] = "Strong Bull"
    band[ok & (composite >= b_bull) & (composite < b_sbull)] = "Bull"
    band[ok & (composite >= b_neut) & (composite < b_bull)] = "Neutral"
    band[ok & (composite >= b_bear) & (composite < b_neut)] = "Bear"
    band[ok & (composite < b_bear)] = "Strong Bear"
    out["EMA_trend"] = band

    # ---- completeness marker: how much of the ribbon actually backs the score --
    # With short history the slower EMAs are NaN and the weighted blend simply
    # renormalizes over what exists -- so a 150-row symbol can print
    # "EMA_score 96.6 / Strong Bull" off 3 of 4 EMAs and look identical in the
    # CSV to a fully-qualified reading. That is fine for a single chart but wrong
    # for cross-sectional ranking, so record the basis explicitly.
    _ema_avail = np.zeros(n)
    for p in EMA_PERIODS:
        _ema_avail += (~np.isnan(emas[p])).astype(float)
    out["EMA_ribbon_n"] = _ema_avail.astype(int)
    _full = len(EMA_PERIODS)
    _eb = np.where(_ema_avail >= _full, "full",
                   np.where(_ema_avail > 0, "partial", ""))
    _eb = np.where(np.isnan(composite), "", _eb)
    out["EMA_score_basis"] = _eb

    out = out.copy()          # defragment before the next column batch
    # ================= SUPERTREND derived fields ============================
    mask("ST_line", st_len)
    mask("ST_dir", st_len)
    st_line = out["ST_line"].to_numpy(dtype=float)
    st_dir  = out["ST_dir"].to_numpy(dtype=float)

    # normalized distance from the Supertrend line -> comparable across stocks
    with np.errstate(invalid="ignore", divide="ignore"):
        out["ST_dist_pct"] = np.round((close_np - st_line) / st_line * 100.0, 4)

    # readable state + flip signals
    st_state = np.full(n, "", dtype=object)
    st_sig   = np.full(n, "", dtype=object)
    prev_dir = np.concatenate([[np.nan], st_dir[:-1]])
    for i in range(n):
        if np.isnan(st_dir[i]):
            continue
        st_state[i] = "Uptrend" if st_dir[i] > 0 else "Downtrend"
        if not np.isnan(prev_dir[i]) and st_dir[i] != prev_dir[i]:
            st_sig[i] = "Buy" if st_dir[i] > 0 else "Sell"
        else:
            st_sig[i] = "Hold"
    out["ST_state"]  = st_state
    out["ST_signal"] = st_sig

    # bars since the last flip -> trend maturity
    out["ST_bars_since_flip"] = _bars_since(np.isin(st_sig, ("Buy", "Sell")))

    out = out.copy()          # defragment before the next column batch
    # ================= MARKET STRUCTURE (swing / pivot based) ===============
    # Pivots are CONFIRMED ONLY: a swing high at bar i requires `sn` higher bars
    # on BOTH sides, so it is only known at bar i+sn. We record it at i+sn (not i)
    # to avoid lookahead bias -- essential if these feed an ML model.
    sn = int(cfg["ms_swing_n"])
    tol = float(cfg["ms_range_tol_pct"]) / 100.0
    high_np = out["High"].to_numpy(dtype=float)
    low_np  = out["Low"].to_numpy(dtype=float)

    sw_high = np.full(n, np.nan)   # confirmed swing high price (at confirm bar)
    sw_low  = np.full(n, np.nan)
    for i in range(sn, n - sn):
        w_hi = high_np[i - sn:i + sn + 1]
        w_lo = low_np[i - sn:i + sn + 1]
        if high_np[i] == np.nanmax(w_hi) and np.sum(w_hi == high_np[i]) == 1:
            if i + sn < n:
                sw_high[i + sn] = high_np[i]
        if low_np[i] == np.nanmin(w_lo) and np.sum(w_lo == low_np[i]) == 1:
            if i + sn < n:
                sw_low[i + sn] = low_np[i]
    out["MS_swing_high"] = np.round(sw_high, 4)
    out["MS_swing_low"]  = np.round(sw_low, 4)

    # walk forward classifying each new swing as HH/LH (highs) or HL/LL (lows)
    last_HH = np.full(n, np.nan); last_LH = np.full(n, np.nan)
    last_HL = np.full(n, np.nan); last_LL = np.full(n, np.nan)
    ms_struct = np.full(n, "", dtype=object)
    ms_event  = np.full(n, "", dtype=object)
    ms_bias   = np.full(n, "", dtype=object)

    prev_high = np.nan; prev_low = np.nan
    cur_HH = cur_LH = cur_HL = cur_LL = np.nan
    # bar index at which each swing type most recently printed (-1 = never)
    idx_HH = idx_LH = idx_HL = idx_LL = -1
    bias = ""                     # running bias, flips on CHoCH
    for i in range(n):
        # --- new confirmed swing high ---
        if not np.isnan(sw_high[i]):
            h = sw_high[i]
            if not np.isnan(prev_high):
                if h > prev_high * (1 + tol):
                    cur_HH = h; idx_HH = i
                    # breaking prior high while bearish = change of character
                    ms_event[i] = "CHoCH Bullish" if bias == "Bearish" else "BOS Bullish"
                    bias = "Bullish"
                elif h < prev_high * (1 - tol):
                    cur_LH = h; idx_LH = i
            prev_high = h
        # --- new confirmed swing low ---
        if not np.isnan(sw_low[i]):
            l = sw_low[i]
            if not np.isnan(prev_low):
                if l < prev_low * (1 - tol):
                    cur_LL = l; idx_LL = i
                    ms_event[i] = "CHoCH Bearish" if bias == "Bullish" else "BOS Bearish"
                    bias = "Bearish"
                elif l > prev_low * (1 + tol):
                    cur_HL = l; idx_HL = i
            prev_low = l

        last_HH[i], last_LH[i] = cur_HH, cur_LH
        last_HL[i], last_LL[i] = cur_HL, cur_LL
        ms_bias[i] = bias

        # Structure from the RECENCY of swing types, not just their existence.
        # (A stale HH from months ago must not keep labelling a broken market
        # as an uptrend.) We track the bar index at which each type last printed
        # and compare: newest high-type vs newest low-type.
        hi_up   = idx_HH   # bar of last Higher High
        hi_down = idx_LH   # bar of last Lower High
        lo_up   = idx_HL   # bar of last Higher Low
        lo_down = idx_LL   # bar of last Lower Low

        high_bull = (hi_up  > hi_down) if (hi_up >= 0 or hi_down >= 0) else None
        low_bull  = (lo_up  > lo_down) if (lo_up >= 0 or lo_down >= 0) else None

        if high_bull is None and low_bull is None:
            ms_struct[i] = ""
        elif high_bull is True and low_bull is True:
            ms_struct[i] = "Uptrend (HH+HL)"
        elif high_bull is False and low_bull is False:
            ms_struct[i] = "Downtrend (LH+LL)"
        else:
            ms_struct[i] = "Range"

    out["MS_last_HH"] = np.round(last_HH, 4)
    out["MS_last_HL"] = np.round(last_HL, 4)
    out["MS_last_LH"] = np.round(last_LH, 4)
    out["MS_last_LL"] = np.round(last_LL, 4)
    out["MS_structure"] = ms_struct
    out["MS_event"] = ms_event
    out["MS_bias"] = ms_bias

    out = out.copy()          # defragment before the next column batch
    # ================= VOLUME derived / interpretive fields =================
    mask("Vol_SMA_fast", v_fast); mask("Vol_SMA_slow", v_slow)
    mask("Vol_ratio", v_fast); mask("VWAP_roll", rw)
    mask("px_vs_VWAP_roll_pct", rw)
    mask("VWMA", int(cfg["vwma_length"]))
    mask("CMF", int(cfg["cmf_length"])); mask("MFI", int(cfg["mfi_length"]))
    mask("EFI", int(cfg["efi_length"])); mask("EOM", int(cfg["eom_length"]))
    mask("ADOSC", int(cfg["adosc_slow"]))

    # OBV slope: % change over lookback -> is accumulation actually rising?
    obv_lb = int(cfg["obv_slope_lb"])
    obv_s = out["OBV"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["OBV_slope"] = (obv_s - obv_s.shift(obv_lb)) / obv_s.shift(obv_lb).abs() * 100.0

    spike_m = float(cfg["vol_spike_mult"]); dry_m = float(cfg["vol_dry_mult"])
    vr = out["Vol_ratio"].to_numpy(dtype=float)

    # Vol_spike: is today's volume unusually heavy?
    vs = np.full(n, "", dtype=object)
    ok_vr = ~np.isnan(vr)
    vs[ok_vr] = "No"
    vs[ok_vr & (vr >= spike_m)] = "Yes"
    out["Vol_spike"] = vs

    # Vol_state: heavy / normal / dry-up, from the stock's OWN average
    vstate = np.full(n, "", dtype=object)
    vstate[ok_vr & (vr >= spike_m)] = "High"
    vstate[ok_vr & (vr < spike_m) & (vr > dry_m)] = "Normal"
    vstate[ok_vr & (vr <= dry_m)] = "Dry-up"
    out["Vol_state"] = vstate

    # Vol_trend: is the short volume average above the long one?
    vf = out["Vol_SMA_fast"].to_numpy(dtype=float)
    vsl = out["Vol_SMA_slow"].to_numpy(dtype=float)
    vt = np.full(n, "", dtype=object)
    ok_vt = (~np.isnan(vf)) & (~np.isnan(vsl))
    vt[ok_vt & (vf > vsl)] = "Rising"
    vt[ok_vt & (vf <= vsl)] = "Falling"
    out["Vol_trend"] = vt

    # Vol_price_confirm: does volume CONFIRM the price move?
    # A price move on heavy volume is confirmed; on light volume it is suspect
    # (classic divergence warning). Uses same-day % change vs relative volume.
    pct_chg = out["Close"].astype(float).pct_change() * 100.0
    pc = pct_chg.to_numpy()
    conf = np.full(n, "", dtype=object)
    okc = ok_vr & (~np.isnan(pc))
    up = okc & (pc > 0); dn = okc & (pc < 0)
    heavy = vr >= 1.0
    conf[up & heavy]  = "Confirmed Up"
    conf[up & ~heavy] = "Weak Up"
    conf[dn & heavy]  = "Confirmed Down"
    conf[dn & ~heavy] = "Weak Down"
    conf[okc & (pc == 0)] = "Flat"
    out["Vol_price_confirm"] = conf

    # MFI_state: overbought / oversold on the volume-weighted RSI
    mfi_u = float(cfg["mfi_upper"]); mfi_l = float(cfg["mfi_lower"])
    mfi = out["MFI"].to_numpy(dtype=float)
    ms_ = np.full(n, "", dtype=object)
    okm = ~np.isnan(mfi)
    ms_[okm] = "Neutral"
    ms_[okm & (mfi >= mfi_u)] = "Overbought"
    ms_[okm & (mfi <= mfi_l)] = "Oversold"
    out["MFI_state"] = ms_

    # CMF_state: net buying vs selling pressure (CMF is bounded roughly -1..1)
    cmf = out["CMF"].to_numpy(dtype=float)
    cs = np.full(n, "", dtype=object)
    okcm = ~np.isnan(cmf)
    cs[okcm & (cmf > 0.05)]  = "Accumulation"
    cs[okcm & (cmf < -0.05)] = "Distribution"
    cs[okcm & (cmf >= -0.05) & (cmf <= 0.05)] = "Neutral"
    out["CMF_state"] = cs

    out = out.copy()          # defragment before the next column batch
    # ============ BB / ADX / STOCHASTIC DECISION FRAMEWORK ==================
    # Core idea: ADX is the GATEKEEPER. It gives no direction, but it tells you
    # WHICH other indicator to trust. Stochastic works in ranges and fails in
    # trends; Bollinger Bands mean-revert in ranges but are ridden in trends.
    # So regime is resolved first, then BB/Stoch are read inside that regime.

    bbu = out["BB_upper"].astype(float); bbl = out["BB_lower"].astype(float)
    bbm = out["BB_mid"].astype(float)
    cl_f = out["Close"].astype(float)

    # ---- Bollinger: %B (position) and bandwidth (volatility state) ----
    with np.errstate(invalid="ignore", divide="ignore"):
        out["BB_pctB"] = (cl_f - bbl) / (bbu - bbl)
        out["BB_bandwidth"] = (bbu - bbl) / bbm * 100.0

    sq_win = min(int(cfg["bb_squeeze_win"]), n)
    sq_pctl = float(cfg["bb_squeeze_pctl"])
    bw = out["BB_bandwidth"]
    bw_rank = bw.rolling(sq_win, min_periods=max(20, sq_win // 4)).rank(pct=True) * 100.0
    out["BB_bandwidth_pctl"] = bw_rank.round(2)

    bwr = bw_rank.to_numpy(dtype=float)
    sq = np.full(n, "", dtype=object)
    ok_bw = ~np.isnan(bwr)
    sq[ok_bw] = "No"
    sq[ok_bw & (bwr <= sq_pctl)] = "Yes"
    out["BB_squeeze"] = sq

    pb_lo = float(cfg["bb_pctb_low"]); pb_hi = float(cfg["bb_pctb_high"])
    pctb = out["BB_pctB"].to_numpy(dtype=float)
    bb_state = np.full(n, "", dtype=object)
    okb = ~np.isnan(pctb)
    bb_state[okb & (pctb >= 1.0)] = "Above Upper"
    bb_state[okb & (pctb >= pb_hi) & (pctb < 1.0)] = "At Upper"
    bb_state[okb & (pctb > 0.5) & (pctb < pb_hi)] = "Upper Half"
    bb_state[okb & (pctb <= 0.5) & (pctb > pb_lo)] = "Lower Half"
    bb_state[okb & (pctb <= pb_lo) & (pctb > 0.0)] = "At Lower"
    bb_state[okb & (pctb <= 0.0)] = "Below Lower"
    out["BB_state"] = bb_state

    # ---- ADX: regime, direction (from DI), and whether it is rising ----
    a_no = float(cfg["adx_no_trend"]); a_tr = float(cfg["adx_trending"])
    a_st = float(cfg["adx_strong"]);   a_ex = float(cfg["adx_extreme"])
    adx_v = out["ADX_14"].to_numpy(dtype=float)
    regime = np.full(n, "", dtype=object)
    oka = ~np.isnan(adx_v)
    regime[oka & (adx_v < a_no)] = "No Trend"
    regime[oka & (adx_v >= a_no) & (adx_v < a_tr)] = "Emerging"
    regime[oka & (adx_v >= a_tr) & (adx_v < a_st)] = "Trending"
    regime[oka & (adx_v >= a_st) & (adx_v < a_ex)] = "Strong"
    regime[oka & (adx_v >= a_ex)] = "Extreme"
    out["ADX_regime"] = regime

    dip = out["DI_plus"].to_numpy(dtype=float)
    dim = out["DI_minus"].to_numpy(dtype=float)
    adir = np.full(n, "", dtype=object)
    okd = (~np.isnan(dip)) & (~np.isnan(dim))
    adir[okd & (dip > dim)] = "Bullish"
    adir[okd & (dip <= dim)] = "Bearish"
    out["ADX_direction"] = adir

    a_lb = int(cfg["adx_slope_lb"])
    adx_chg = out["ADX_14"].astype(float).diff(a_lb).to_numpy()
    arise = np.full(n, "", dtype=object)
    okr = ~np.isnan(adx_chg)
    arise[okr & (adx_chg > 0)] = "Yes"
    arise[okr & (adx_chg <= 0)] = "No"
    out["ADX_rising"] = arise

    # ================= COMPOSITE TREND SCORE ================================
    # Blends four independent views of trend into one 0-100 number:
    #   EMA ribbon      -- where price sits in its own moving-average structure
    #   Supertrend      -- volatility-anchored direction and distance
    #   ADX (+DI/-DI)   -- how STRONG the trend is, signed by direction
    #   Market structure-- the swing-high/swing-low sequence
    #
    # They are deliberately different lenses: EMA is smooth and laggy, Supertrend
    # is binary and volatility-based, ADX measures conviction without direction,
    # and structure is discrete price action. When they agree the read is robust;
    # when they disagree that disagreement is itself the signal, which is why
    # Trend_agreement is reported alongside the score rather than hidden inside it.
    #
    # Every sub-score is mapped to 0-100 where 50 = neutral, so the weights are
    # comparable. Sub-scores are exposed as their own columns so a surprising
    # Trend_score can always be decomposed.

    # ---- 1. EMA sub-score: already 0-100 from the ribbon composite ----------
    _ts_ema = out["EMA_score"].to_numpy(dtype=float)

    # ---- 2. Supertrend sub-score: direction plus graded distance -----------
    # Direction sets the half (above 50 = up), distance from the line grades it.
    # Saturating means a runaway gap cannot swamp the other components.
    _st_sat = float(cfg["trend_st_saturate_pct"])
    _st_d = out["ST_dir"].to_numpy(dtype=float)
    _st_dist = out["ST_dist_pct"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        _st_mag = np.clip(np.abs(_st_dist) / _st_sat, 0.0, 1.0)
    _ts_st = np.full(n, np.nan)
    _okst = (~np.isnan(_st_d)) & (~np.isnan(_st_mag))
    # up: 50 -> 100 as distance grows; down: 50 -> 0
    _ts_st[_okst] = 50.0 + np.sign(_st_d[_okst]) * 50.0 * _st_mag[_okst]

    # ---- 3. ADX sub-score: strength SIGNED by DI ---------------------------
    # ADX is directionless -- a violent downtrend has a high ADX. Feeding the
    # raw value in would score a crash as bullish, so it is signed by DI and
    # damped below the floor where ADX indicates chop rather than trend.
    _adx_sat = float(cfg["trend_adx_saturate"])
    _adx_flr = float(cfg["trend_adx_floor"])
    with np.errstate(invalid="ignore"):
        _adx_mag = np.clip((adx_v - _adx_flr) / max(_adx_sat - _adx_flr, 1e-9), 0.0, 1.0)
    _di_sign = np.where(np.isnan(dip) | np.isnan(dim), np.nan,
                        np.where(dip > dim, 1.0, -1.0))
    _ts_adx = np.full(n, np.nan)
    _okadx = (~np.isnan(_adx_mag)) & (~np.isnan(_di_sign))
    _ts_adx[_okadx] = 50.0 + _di_sign[_okadx] * 50.0 * _adx_mag[_okadx]

    # ---- 4. Market structure sub-score: discrete -> 0 / 50 / 100 -----------
    _ts_ms = np.full(n, np.nan)
    _ms_arr = np.asarray(ms_struct, dtype=object)
    _ts_ms[_ms_arr == "Uptrend (HH+HL)"]   = 100.0
    _ts_ms[_ms_arr == "Range"]             = 50.0
    _ts_ms[_ms_arr == "Downtrend (LH+LL)"] = 0.0

    # Trend_ema_sub is not emitted: it is EMA_score verbatim. The blend below
    # still uses _ts_ema; duplicating it as a column adds width, not information.
    out["Trend_supertrend_sub"] = np.round(_ts_st, 2)
    out["Trend_adx_sub"]       = np.round(_ts_adx, 2)
    out["Trend_structure_sub"] = np.round(_ts_ms, 2)

    # ---- weighted blend, renormalized over available components ------------
    _TW = {"ema": float(cfg["trend_w_ema"]),
           "supertrend": float(cfg["trend_w_supertrend"]),
           "adx": float(cfg["trend_w_adx"]),
           "structure": float(cfg["trend_w_structure"])}
    if sum(_TW.values()) <= 0:
        print("  config: all trend_w_* weights are 0 -> Trend_score left blank")

    # Suppress bars backed by too few components: an early score built on the
    # fastest EMA alone is not comparable to a full four-component reading.
    _trend, _t_navail, _tb = _weighted_blend(
        n, {"ema": _ts_ema, "supertrend": _ts_st, "adx": _ts_adx, "structure": _ts_ms},
        _TW, int(cfg["trend_min_components"]))
    out["Trend_score"] = np.round(_trend, 2)
    out["Trend_components_n"] = _t_navail
    out["Trend_score_basis"] = _tb

    # ---- readable band -----------------------------------------------------
    _tsb = float(cfg["trend_band_strong_bull"]); _tb_b = float(cfg["trend_band_bull"])
    _tn = float(cfg["trend_band_neutral"]);      _tbe = float(cfg["trend_band_bear"])
    _tband = np.full(n, "", dtype=object)
    _okt = ~np.isnan(_trend)
    _tband[_okt & (_trend >= _tsb)] = "Strong Bull"
    _tband[_okt & (_trend >= _tb_b) & (_trend < _tsb)] = "Bull"
    _tband[_okt & (_trend >= _tn) & (_trend < _tb_b)] = "Neutral"
    _tband[_okt & (_trend >= _tbe) & (_trend < _tn)] = "Bear"
    _tband[_okt & (_trend < _tbe)] = "Strong Bear"
    out["Trend_band"] = _tband

    # ---- agreement: do the components actually point the same way? ---------
    # A 62 built from four components all mildly bullish is a very different
    # situation from a 62 built from two strongly bullish and two bearish ones.
    # The score alone cannot distinguish them; this can.
    _bull_n = np.zeros(n); _bear_n = np.zeros(n)
    for _arr in (_ts_ema, _ts_st, _ts_adx, _ts_ms):
        _v = ~np.isnan(_arr)
        _bull_n += np.where(_v & (_arr > 50.0), 1, 0)
        _bear_n += np.where(_v & (_arr < 50.0), 1, 0)
    out["Trend_bull_components"] = _bull_n.astype(int)
    out["Trend_bear_components"] = _bear_n.astype(int)

    _amin = int(cfg["trend_agree_min"])
    _agree = np.full(n, "", dtype=object)
    # Gate on _okt (Trend_score present), not just component availability, so
    # agreement never appears on a bar whose score was suppressed.
    _agree[_okt] = "Mixed"
    _agree[_okt & (_bull_n >= _amin) & (_bear_n == 0)] = "Bullish confirmed"
    _agree[_okt & (_bear_n >= _amin) & (_bull_n == 0)] = "Bearish confirmed"
    _agree[_okt & (_bull_n >= _amin) & (_bear_n > 0)] = "Bullish majority"
    _agree[_okt & (_bear_n >= _amin) & (_bull_n > 0)] = "Bearish majority"
    out["Trend_agreement"] = _agree

    # ---- description: the one-line human read ------------------------------
    # Band says WHAT, agreement says HOW RELIABLE, ADX regime says whether a
    # trend exists at all. Order matters: a weak ADX regime OVERRIDES rather
    # than decorates, because "Strong Bull - all components agree (no trend
    # strength)" is self-contradictory. Directional bias without ADX
    # confirmation is a range with a lean, and the text should say exactly that.
    _weak_regime = {"No Trend", "Emerging"}
    _desc = np.full(n, "", dtype=object)
    for i in range(n):
        if not _okt[i]:
            continue
        b = _tband[i]; ag = _agree[i]; rg = regime[i]
        lean = ("bullish" if b in ("Strong Bull", "Bull")
                else "bearish" if b in ("Strong Bear", "Bear") else "flat")
        if rg in _weak_regime:
            if lean == "flat":
                _desc[i] = f"No clear trend - ADX {rg.lower()}, price ranging"
            else:
                _desc[i] = (f"Range with {lean} lean - ADX {rg.lower()}, "
                            f"direction not confirmed")
        elif ag in ("Bullish confirmed", "Bearish confirmed"):
            _desc[i] = f"{b} - all {int(_t_navail[i])} components agree, {rg.lower()} ADX"
        elif ag in ("Bullish majority", "Bearish majority"):
            _dis = int(_bear_n[i] if "Bull" in ag else _bull_n[i])
            _desc[i] = f"{b} - majority agree, {_dis} dissenting, {rg.lower()} ADX"
        else:
            _desc[i] = f"{b} - components mixed, low conviction"
    out["Trend_description"] = _desc

    out = out.copy()          # defragment after the trend batch

    # ---- Stochastic: state + crossover ----
    s_up = float(cfg["stoch_upper"]); s_lo = float(cfg["stoch_lower"])
    k = out["Stoch_K"].to_numpy(dtype=float)
    d_ = out["Stoch_D"].to_numpy(dtype=float)
    st_state2 = np.full(n, "", dtype=object)
    oks = ~np.isnan(k)
    st_state2[oks] = "Neutral"
    st_state2[oks & (k >= s_up)] = "Overbought"
    st_state2[oks & (k <= s_lo)] = "Oversold"
    out["Stoch_state"] = st_state2

    kp = np.concatenate([[np.nan], k[:-1]])
    dp = np.concatenate([[np.nan], d_[:-1]])
    cross = np.full(n, "", dtype=object)
    okx = (~np.isnan(k)) & (~np.isnan(d_)) & (~np.isnan(kp)) & (~np.isnan(dp))
    cross[okx] = "None"
    cross[okx & (kp <= dp) & (k > d_)] = "Bullish Cross"
    cross[okx & (kp >= dp) & (k < d_)] = "Bearish Cross"
    out["Stoch_cross"] = cross

    # ---- ATR: volatility regime + risk / stop placement ----
    # ATR is NOT directional. Its role in a decision framework is (a) telling you
    # whether volatility is expanding or contracting, and (b) sizing the trade:
    # stop distance and therefore position size come from ATR, not from a fixed %.
    # Raw ATR is in price units so it cannot be compared across stocks -- ATR_pct
    # is the normalized form to rank on.
    atr_v = out["ATR_14"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["ATR_pct"] = (atr_v / cl_f * 100.0).round(4)

    ap_win = min(int(cfg["atr_pctl_win"]), n)
    atr_rank = out["ATR_pct"].rolling(
        ap_win, min_periods=max(20, ap_win // 4)).rank(pct=True) * 100.0
    out["ATR_pctl"] = atr_rank.round(2)

    a_lo = float(cfg["atr_low_pctl"]); a_hi = float(cfg["atr_high_pctl"])
    a_exq = float(cfg["atr_extreme_pctl"])
    arv = atr_rank.to_numpy(dtype=float)
    aregime = np.full(n, "", dtype=object)
    okar = ~np.isnan(arv)
    aregime[okar & (arv < a_lo)] = "Low Vol"
    aregime[okar & (arv >= a_lo) & (arv < a_hi)] = "Normal Vol"
    aregime[okar & (arv >= a_hi) & (arv < a_exq)] = "High Vol"
    aregime[okar & (arv >= a_exq)] = "Extreme Vol"
    out["ATR_regime"] = aregime

    at_lb = int(cfg["atr_slope_lb"])
    atr_chg = out["ATR_pct"].astype(float).diff(at_lb).to_numpy()
    arise2 = np.full(n, "", dtype=object)
    okat = ~np.isnan(atr_chg)
    arise2[okat & (atr_chg > 0)] = "Yes"
    arise2[okat & (atr_chg <= 0)] = "No"
    out["ATR_rising"] = arise2

    # stop levels & risk per unit -> feeds position sizing
    st_mult = float(cfg["atr_stop_mult"])
    risk = atr_v * st_mult
    # Risk_per_unit is not emitted: it is exactly atr_stop_mult * ATR_14, and
    # the multiplier is in config. Stop_long/Stop_short carry the usable levels.
    out["Stop_long"]  = (cl_f - risk).round(4)
    out["Stop_short"] = (cl_f + risk).round(4)
    with np.errstate(invalid="ignore", divide="ignore"):
        # Kept as a local for the ATR_note logic below, but not emitted:
        # it is exactly atr_stop_mult * ATR_pct.
        _stop_long_pct = (risk / cl_f * 100.0).round(4)

    # ---- SETUP: the combined, regime-aware read ----
    # This is the payoff column: it encodes the decision table so you do not have
    # to eyeball three indicators per stock. Order matters -- the first matching
    # rule wins, most specific first.
    pl_lo = float(cfg["stoch_pullback_lo"]); pl_hi = float(cfg["stoch_pullback_hi"])
    setup = np.full(n, "", dtype=object)
    # A cross rarely lands on the exact bar price enters the pullback zone, so we
    # allow it within a short window -- same-bar-only reduced valid setups by ~87%.
    xw = int(cfg["stoch_cross_window"])
    cross_recent = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - xw + 1)
        cross_recent[i] = any(cross[j] == "Bullish Cross" for j in range(lo, i + 1))
    for i in range(n):
        if not (oka[i] and okb[i] and oks[i]):
            continue
        reg = regime[i]; bbs = bb_state[i]; sc = cross[i]
        ki = k[i]; bull = (adir[i] == "Bullish"); rising = (arise[i] == "Yes")

        # --- RANGE regime: mean-reversion tools are valid ---
        if reg in ("No Trend", "Emerging"):
            if sq[i] == "Yes":
                setup[i] = "Squeeze Watch"          # breakout pending, no direction yet
            elif bbs in ("At Lower", "Below Lower") and (
                    sc == "Bullish Cross" or ki <= s_lo):
                setup[i] = "Range Long"
            elif bbs in ("At Upper", "Above Upper") and (
                    sc == "Bearish Cross" or ki >= s_up):
                setup[i] = "Range Short"
            else:
                setup[i] = "Range Neutral"
        # --- TRENDING regime: overbought/oversold is NOT a fade signal ---
        else:
            if reg == "Extreme" or (reg == "Strong" and ki >= 90):
                setup[i] = "Extended"              # trail stops, do not initiate
            elif bull and pl_lo <= ki <= pl_hi and cross_recent[i]:
                setup[i] = "Trend Pullback Entry"  # best trend entry
            elif bull and bbs in ("At Upper", "Above Upper"):
                setup[i] = "Trend Intact"          # riding the band = continuation
            elif (not bull) and bbs in ("At Lower", "Below Lower"):
                setup[i] = "Downtrend Intact"
            elif not rising:
                setup[i] = "Trend Decaying"
            else:
                setup[i] = "Trend Neutral"
    out["Setup"] = setup

    # ---- ATR_note: how volatility QUALIFIES the setup ----
    # The same setup is not equally tradeable in every volatility regime. Low vol
    # + squeeze = coiled spring; extreme vol = stops must be so wide the position
    # gets tiny; expanding vol after a squeeze = the breakout is underway.
    wide_thr = float(cfg["atr_wide_stop_pct"])
    slp = _stop_long_pct.to_numpy(dtype=float)
    note = np.full(n, "", dtype=object)
    for i in range(n):
        if not okar[i]:
            continue
        reg = aregime[i]
        if reg == "Extreme Vol":
            note[i] = "Extreme vol - reduce size"
        elif not np.isnan(slp[i]) and slp[i] > wide_thr:
            note[i] = "Wide stop - size down"
        elif reg == "Low Vol" and sq[i] == "Yes":
            note[i] = "Coiled - low vol squeeze"
        elif reg == "Low Vol":
            note[i] = "Low vol - tight stops"
        elif arise2[i] == "Yes" and reg == "High Vol":
            note[i] = "Vol expanding"
        elif arise2[i] == "No" and reg in ("High Vol", "Normal Vol"):
            note[i] = "Vol contracting"
        else:
            note[i] = "Normal vol"
    out["ATR_note"] = note

    out = out.copy()          # defragment before the next column batch
    # ================= STAGE ANALYSIS (Weinstein 4-stage) ===================
    # Stage 1 Basing / 2 Advancing / 3 Topping / 4 Declining.
    # Classification uses price vs a long MA plus that MA's SLOPE (rising / flat /
    # falling). Weinstein used the 30-week MA on WEEKLY bars; with stage_weekly=1
    # we resample to weekly, classify there, then map the label back onto daily
    # rows (forward-fill) -- this is closer to the original method and much less
    # noisy than classifying raw daily bars.
    st_ma_len   = int(cfg["stage_ma_length"])
    st_ma_type  = str(cfg["stage_ma_type"]).upper().strip()
    st_slope_lb = int(cfg["stage_slope_lb"])
    st_flat     = float(cfg["stage_flat_pct"])
    st_min_days = int(cfg["stage_min_days"])
    st_weekly   = int(float(cfg["stage_weekly"])) == 1

    dates = pd.to_datetime(out["Date"])
    close_s = out["Close"].astype(float)

    def _classify(close_ser, ma_len, slope_lb, flat_thr):
        """Return an integer stage series (1-4, NaN before warmup)."""
        if st_ma_type == "EMA":
            ma = close_ser.ewm(span=ma_len, adjust=False, min_periods=ma_len).mean()
        else:
            ma = close_ser.rolling(ma_len).mean()
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = (ma / ma.shift(slope_lb) - 1.0) * 100.0
        above = close_ser > ma
        rising  = slope >  flat_thr
        falling = slope < -flat_thr
        flat    = (~rising) & (~falling) & slope.notna()

        raw = pd.Series(np.nan, index=close_ser.index)
        raw[above & rising]   = 2      # advancing
        raw[above & flat]     = 3      # topping
        raw[~above & flat]    = 1      # basing
        raw[~above & falling] = 4      # declining
        # --- transition cells ---
        # Price ABOVE a falling MA: the advance is rolling over -> topping.
        raw[above & falling]  = 3
        # Price BELOW a RISING MA: this is a pullback inside an uptrend (or the
        # very start of a top) -- NOT basing. Basing requires the MA to have
        # stopped rising. Label it 3 (topping//transition) so a stale uptrend is
        # never mislabelled as accumulation.
        raw[(~above) & rising] = 3
        raw[ma.isna() | slope.isna()] = np.nan
        return raw, ma, slope

    if st_weekly:
        wk = pd.DataFrame({"Date": dates, "Close": close_s}).set_index("Date")
        wk_close = wk["Close"].resample("W-FRI").last().dropna()
        wk_len = max(4, int(round(st_ma_len / 5)))          # 150d -> 30w
        wk_slope_lb = max(2, int(round(st_slope_lb / 5)))
        raw_w, ma_w, slope_w = _classify(wk_close, wk_len, wk_slope_lb, st_flat)
        # map weekly labels back onto daily rows (as-of / forward fill)
        raw_daily = raw_w.reindex(
            pd.DatetimeIndex(dates), method="ffill").to_numpy(dtype=float)
        ma_daily = ma_w.reindex(
            pd.DatetimeIndex(dates), method="ffill").to_numpy(dtype=float)
        slope_daily = slope_w.reindex(
            pd.DatetimeIndex(dates), method="ffill").to_numpy(dtype=float)
    else:
        raw_s, ma_s, slope_s = _classify(close_s, st_ma_len, st_slope_lb, st_flat)
        raw_daily = raw_s.to_numpy(dtype=float)
        ma_daily = ma_s.to_numpy(dtype=float)
        slope_daily = slope_s.to_numpy(dtype=float)

    out["Stage_MA"] = np.round(ma_daily, 4)
    out["Stage_MA_slope"] = np.round(slope_daily, 4)

    # ---- whipsaw filter: a new stage must persist st_min_days to be adopted ----
    stage = np.full(n, np.nan)
    cur = np.nan; pending = np.nan; pending_len = 0
    for i in range(n):
        r = raw_daily[i]
        if np.isnan(r):
            continue
        if np.isnan(cur):
            cur = r; pending = r; pending_len = 0
        elif r == cur:
            pending = r; pending_len = 0
        else:
            if r == pending:
                pending_len += 1
            else:
                pending = r; pending_len = 1
            if pending_len >= st_min_days:
                cur = r; pending_len = 0
        stage[i] = cur

    names = {1: "1 - Basing", 2: "2 - Advancing", 3: "3 - Topping", 4: "4 - Declining"}
    out["Stage"] = stage
    out["Stage_name"] = [names.get(x, "") if not np.isnan(x) else "" for x in stage]

    # ---- run lengths, start info, previous stage, transitions ----
    stage_days = np.full(n, np.nan)
    stage_start_i = np.full(n, -1, dtype=int)
    prev_stage = np.full(n, np.nan)
    prev_days = np.full(n, np.nan)
    transition = np.full(n, "", dtype=object)

    run_start = -1; last_stage = np.nan
    prev_completed = np.nan; prev_completed_len = np.nan
    for i in range(n):
        s_i = stage[i]
        if np.isnan(s_i):
            continue
        if np.isnan(last_stage) or s_i != last_stage:
            if not np.isnan(last_stage):
                prev_completed = last_stage
                prev_completed_len = i - run_start
                transition[i] = f"{int(last_stage)}->{int(s_i)}"
            run_start = i
            last_stage = s_i
        stage_start_i[i] = run_start
        stage_days[i] = i - run_start + 1
        prev_stage[i] = prev_completed
        prev_days[i] = prev_completed_len

    out["Stage_days"] = stage_days
    out["Stage_transition"] = transition
    out["Prev_stage"] = prev_stage
    out["Prev_stage_name"] = [names.get(x, "") if not np.isnan(x) else "" for x in prev_stage]
    out["Prev_stage_days"] = prev_days

    start_date = np.full(n, "", dtype=object)
    start_price = np.full(n, np.nan)
    stage_ret = np.full(n, np.nan)
    d_str = dates.dt.strftime("%Y-%m-%d").to_numpy()
    cl = close_s.to_numpy()
    for i in range(n):
        si = stage_start_i[i]
        if si >= 0:
            start_date[i] = d_str[si]
            start_price[i] = cl[si]
            if cl[si] != 0:
                stage_ret[i] = (cl[i] / cl[si] - 1.0) * 100.0
    out["Stage_start_date"] = start_date
    out["Stage_start_price"] = np.round(start_price, 4)
    out["Stage_return_pct"] = np.round(stage_ret, 4)

    # ---- maturity: current duration vs this stock's OWN typical duration ----
    # Uses the stock's median completed duration for that stage when enough
    # cycles exist, else the configured fallback (same pattern as RSI_basis).
    typ_fallback = {1: float(cfg["stage_typ_days_1"]), 2: float(cfg["stage_typ_days_2"]),
                    3: float(cfg["stage_typ_days_3"]), 4: float(cfg["stage_typ_days_4"])}
    completed = {1: [], 2: [], 3: [], 4: []}
    maturity = np.full(n, np.nan)
    mat_basis = np.full(n, "", dtype=object)
    seen_stage = np.nan; seen_start = -1
    for i in range(n):
        s_i = stage[i]
        if np.isnan(s_i):
            continue
        if np.isnan(seen_stage):
            seen_stage = s_i; seen_start = i
        elif s_i != seen_stage:
            completed[int(seen_stage)].append(i - seen_start)
            seen_stage = s_i; seen_start = i
        hist = completed[int(s_i)]
        if len(hist) >= 2:
            typ = float(np.median(hist)); basis = "adaptive"
        else:
            typ = typ_fallback[int(s_i)]; basis = "fixed"
        if typ > 0:
            maturity[i] = stage_days[i] / typ * 100.0
        mat_basis[i] = basis
    out["Stage_maturity_pct"] = np.round(maturity, 2)

    early_p = float(cfg["stage_early_pct"]); late_p = float(cfg["stage_late_pct"])
    sub = np.full(n, "", dtype=object)
    okm2 = ~np.isnan(maturity)
    sub[okm2 & (maturity < early_p)] = "early"
    sub[okm2 & (maturity >= early_p) & (maturity <= late_p)] = "mid"
    sub[okm2 & (maturity > late_p)] = "late"
    out["Stage_substage"] = sub

    # ---- confidence: does independent structure agree with the stage call? ----
    conf_thr = float(cfg["stage_confirm_score"])
    ema_struct = out["EMA_structure_score"].to_numpy(dtype=float)
    conf = np.full(n, np.nan)
    for i in range(n):
        s_i = stage[i]
        if np.isnan(s_i) or np.isnan(ema_struct[i]):
            continue
        es = ema_struct[i]
        if s_i == 2:      agree = es                    # advancing wants high structure
        elif s_i == 4:    agree = 100.0 - es            # declining wants low structure
        else:             agree = 100.0 - abs(es - 50.0) * 2.0   # 1/3 want middling
        base = 60.0 if ((s_i == 2 and es >= conf_thr) or
                        (s_i == 4 and es <= 100 - conf_thr)) else 50.0
        conf[i] = np.clip(base * 0.4 + agree * 0.6, 0, 100)
    out["Stage_confidence"] = np.round(conf, 2)
    # Bollinger (length 20)
    for c in ("BB_upper", "BB_mid", "BB_lower"):
        mask(c, bb_len)
    # MACD(12,26,9): slow EMA 26 + 9 signal -> ~34 before signal/hist are meaningful
    mask("MACD", m_slow)
    mask("MACD_signal", m_slow + m_sig)
    mask("MACD_hist", m_slow + m_sig)
    mask("PPO", m_slow)
    # ADX/ATR/Stochastic are masked immediately after being computed, above.
    # Donchian: rolling min_periods gates the window; the shift costs one bar
    _dc_warm = max(dc_up_len, dc_lo_len) + DC_SHIFT
    for _c in ("DC_upper", "DC_lower", "DC_mid", "DC_width_pct", "DC_pct_rank"):
        mask(_c, _dc_warm)
    # ROC(N): needs N prior closes
    for _p in ROC_PERIODS:
        mask(f"ROC_{_p}", _p)
        mask(f"ROC_{_p}_z", _p)
    # Keltner: EMA(20) + ATR(10) -> gated by the slower of the two
    _kc_warm = max(kc_len, kc_atr_len)
    for _c in ("KC_mid", "KC_upper", "KC_lower", "KC_width_pct", "KC_pct_rank"):
        mask(_c, _kc_warm)

    # Defragment: ~100 sequential single-column inserts above leave the frame
    # highly fragmented, which pandas warns about and which slows every further
    # assignment. One copy here consolidates the blocks.
    out = out.copy()

    # --- normalized / readable MACD fields (computed AFTER warmup mask) ---
    # MACD_hist_z : histogram z-scored vs the stock's own history -> extremity of
    #               momentum for THIS stock, comparable across the universe.
    # MACD_state  : human-readable momentum read from MACD line & histogram:
    #               bullish/bearish (line vs 0), accelerating/fading (|hist| trend),
    #               and fresh signal-line crossovers.
    line = out["MACD"]
    hist = out["MACD_hist"]
    win = min(int(cfg["macd_hist_z_win"]), n)
    hroll = hist.rolling(win, min_periods=max(30, win // 4))
    out["MACD_hist_z"] = (hist - hroll.mean()) / hroll.std()

    h = hist.to_numpy()
    l = line.to_numpy()
    hp = np.concatenate([[np.nan], h[:-1]])    # previous histogram
    state = np.full(n, "", dtype=object)
    for i in range(n):
        if np.isnan(h[i]) or np.isnan(l[i]):
            continue
        # fresh crossover (histogram sign flip) takes priority
        if not np.isnan(hp[i]) and hp[i] < 0 <= h[i]:
            state[i] = "Bullish crossover"
        elif not np.isnan(hp[i]) and hp[i] > 0 >= h[i]:
            state[i] = "Bearish crossover"
        else:
            trend = "bullish" if l[i] > 0 else "bearish"
            if not np.isnan(hp[i]) and (h[i] * hp[i] > 0):
                momo = "accelerating" if abs(h[i]) > abs(hp[i]) else "fading"
                state[i] = f"{trend} {momo}"
            else:
                state[i] = trend
    out["MACD_state"] = state

    out = out.copy()          # defragment before the next column batch
    # ============ DONCHIAN / ROC / KELTNER derived fields ===================
    # Computed AFTER the warm-up mask so partial-lookback rows stay blank.
    _c_np = out["Close"].to_numpy(dtype=float)
    _dcu  = out["DC_upper"].to_numpy(dtype=float)
    _dcl  = out["DC_lower"].to_numpy(dtype=float)
    _tol  = float(cfg["donchian_prox_tol"]) / 100.0

    # ---- Donchian breakout state ----
    dc_state = np.full(n, "", dtype=object)
    _dok = (~np.isnan(_dcu)) & (~np.isnan(_dcl)) & (~np.isnan(_c_np))
    dc_state[_dok] = "Inside"
    dc_state[_dok & (_c_np <= _dcu) & (_c_np >= _dcu * (1 - _tol))] = "Near Upper"
    dc_state[_dok & (_c_np >= _dcl) & (_c_np <= _dcl * (1 + _tol))] = "Near Lower"
    # breakouts assigned LAST so they take precedence over the proximity flags
    dc_state[_dok & (_c_np > _dcu)] = "Breakout Up"
    dc_state[_dok & (_c_np < _dcl)] = "Breakout Down"
    out["DC_state"] = dc_state

    # bars since the last breakout -> freshness. Stays NaN until the first one
    # actually occurs (no synthetic zero on row 0).
    _dc_bo = np.isin(dc_state, ["Breakout Up", "Breakout Down"])
    out["DC_bars_since_breakout"] = _bars_since(_dc_bo)

    # ---- ROC z-score trust basis (mirrors the RSI_basis convention) ----
    _roc_full = int(cfg["roc_z_full"])
    for _p in ROC_PERIODS:
        _zc = f"ROC_{_p}_z"
        _isna = out[_zc].isna().to_numpy()
        _enough = (idx >= (_roc_full + _p)) & (~_isna)
        _b = np.where(_enough, "full", "thin").astype(object)
        _b[_isna] = ""
        out[f"{_zc}_basis"] = _b

    # ---- ROC composite score (0-100, 50 = flat) ----
    _rp = int(cfg["roc_score_period"])
    if _rp not in ROC_PERIODS:
        _rp = min(ROC_PERIODS, key=lambda x: abs(x - _rp))
        print(f"  config: roc_score_period not in roc_periods -> using ROC_{_rp}")
    _roc = out[f"ROC_{_rp}"].to_numpy(dtype=float)
    _sat = float(cfg["roc_saturate_pct"])
    with np.errstate(invalid="ignore"):
        out["ROC_score"] = np.round(50.0 + 50.0 * np.clip(_roc / _sat, -1.0, 1.0), 2)

    # direction + 2nd derivative (is the rate of change itself rising?)
    roc_state = np.full(n, "", dtype=object)
    _rok = ~np.isnan(_roc)
    roc_state[_rok & (_roc > 0)]  = "Positive"
    roc_state[_rok & (_roc <= 0)] = "Negative"
    _rprev = np.concatenate([[np.nan], _roc[:-1]])
    _acc = _rok & (~np.isnan(_rprev))
    roc_state[_acc & (_roc > 0)  & (_roc >  _rprev)] = "Positive accelerating"
    roc_state[_acc & (_roc > 0)  & (_roc <= _rprev)] = "Positive fading"
    roc_state[_acc & (_roc <= 0) & (_roc <  _rprev)] = "Negative accelerating"
    roc_state[_acc & (_roc <= 0) & (_roc >= _rprev)] = "Negative fading"
    out["ROC_state"] = roc_state

    # ================= COMPOSITE MOMENTUM SCORE =============================
    # Blends four views of momentum into one 0-100 number:
    #   RSI        -- normalized position of recent gains vs losses
    #   ROC        -- raw rate of change over the headline horizon
    #   MACD hist  -- whether the trend is ACCELERATING or decelerating
    #   Stoch %K   -- where price sits inside its recent high/low range
    #
    # These are NOT independent -- all four are functions of recent returns, so
    # they agree far more often than the four trend components do. That makes
    # agreement here weaker evidence than it looks, and it is why the composite
    # is deliberately paired with Momentum_divergence below: the informative
    # case is momentum disagreeing with PRICE, not the components agreeing with
    # each other.
    #
    # Every sub-score is mapped to 0-100 where 50 = neutral, and each is exposed
    # as its own column so a surprising score can be decomposed.

    # ---- 1. RSI sub-score: already 0-100 -----------------------------------
    _ms_rsi = out["RSI_14"].to_numpy(dtype=float)

    # ---- 2. ROC sub-score: already 0-100 from the ROC block ----------------
    _ms_roc = out["ROC_score"].to_numpy(dtype=float)

    # ---- 3. MACD sub-score: histogram z-score, squashed --------------------
    # The raw histogram is unbounded and scales with price, so it cannot be
    # compared across stocks. MACD_hist_z normalizes it against its own history;
    # tanh then maps it to a bounded 0-100 without letting one spike dominate.
    _mz_sat = float(cfg["mom_macd_z_sat"])
    _mh_z = out["MACD_hist_z"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        _ms_macd = 50.0 + 50.0 * np.tanh(_mh_z / max(_mz_sat, 1e-9))

    # ---- 4. Stochastic sub-score: %K, damped -------------------------------
    # %K is used directly (high = upward momentum), NOT inverted for overbought.
    # In a trend, a pinned-high %K is strength, not a sell signal; the
    # overbought READ belongs in the description, not baked into the score.
    # Damping pulls it toward 50 because %K is the noisiest of the four.
    _sd = float(cfg["mom_stoch_damp"])
    _k_arr = out["Stoch_K"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        _ms_stoch = 50.0 + (_k_arr - 50.0) * _sd

    # Only MACD is emitted as a sub-score. The other three are restatements of
    # columns that already exist: Mom_rsi_sub == RSI_14, Mom_roc_sub ==
    # ROC_score, and Mom_stoch_sub is just 0.8*Stoch_K + 10. The blend still
    # uses all four internally.
    out["Mom_macd_sub"]  = np.round(_ms_macd, 2)

    # ---- weighted blend, renormalized over available components ------------
    _MW = {"rsi": float(cfg["mom_w_rsi"]), "roc": float(cfg["mom_w_roc"]),
           "macd": float(cfg["mom_w_macd"]), "stoch": float(cfg["mom_w_stoch"])}
    _mom, _m_navail, _mb = _weighted_blend(
        n, {"rsi": _ms_rsi, "roc": _ms_roc, "macd": _ms_macd, "stoch": _ms_stoch},
        _MW, int(cfg["mom_min_components"]))
    out["Momentum_score"] = np.round(_mom, 2)
    out["Momentum_score_basis"] = _mb

    # ---- readable band -----------------------------------------------------
    _msb = float(cfg["mom_band_strong_bull"]); _mb_b = float(cfg["mom_band_bull"])
    _mn = float(cfg["mom_band_neutral"]);      _mbe = float(cfg["mom_band_bear"])
    _mband = np.full(n, "", dtype=object)
    _okm = ~np.isnan(_mom)
    _mband[_okm & (_mom >= _msb)] = "Strong Bullish"
    _mband[_okm & (_mom >= _mb_b) & (_mom < _msb)] = "Bullish"
    _mband[_okm & (_mom >= _mn) & (_mom < _mb_b)] = "Neutral"
    _mband[_okm & (_mom >= _mbe) & (_mom < _mn)] = "Bearish"
    _mband[_okm & (_mom < _mbe)] = "Strong Bearish"
    out["Momentum_band"] = _mband

    # ---- agreement ---------------------------------------------------------
    _mbull = np.zeros(n); _mbear = np.zeros(n)
    for _arr in (_ms_rsi, _ms_roc, _ms_macd, _ms_stoch):
        _v = ~np.isnan(_arr)
        _mbull += np.where(_v & (_arr > 50.0), 1, 0)
        _mbear += np.where(_v & (_arr < 50.0), 1, 0)
    out["Momentum_bull_components"] = _mbull.astype(int)
    out["Momentum_bear_components"] = _mbear.astype(int)

    # ---- effective independence -------------------------------------------
    # RSI, ROC and Stochastic are all monotone functions of recent returns and
    # typically correlate above 0.85 with each other; MACD histogram is the only
    # component measuring something different (acceleration). So "all 4 agree"
    # frequently means one signal counted four times. This field reports how
    # many DISTINCT views actually back the reading: the return-based family
    # counts once, MACD counts once. Use it to discount the agreement label.
    _fam_ret = np.zeros(n, dtype=bool)      # RSI / ROC / Stoch family present
    for _arr in (_ms_rsi, _ms_roc, _ms_stoch):
        _fam_ret |= ~np.isnan(_arr)
    _fam_acc = ~np.isnan(_ms_macd)          # MACD acceleration family
    out["Momentum_distinct_views"] = (_fam_ret.astype(int) + _fam_acc.astype(int))

    _magr = int(cfg["mom_agree_min"])
    _magree = np.full(n, "", dtype=object)
    _magree[_okm] = "Mixed"
    _magree[_okm & (_mbull >= _magr) & (_mbear == 0)] = "Bullish confirmed"
    _magree[_okm & (_mbear >= _magr) & (_mbull == 0)] = "Bearish confirmed"
    _magree[_okm & (_mbull >= _magr) & (_mbear > 0)] = "Bullish majority"
    _magree[_okm & (_mbear >= _magr) & (_mbull > 0)] = "Bearish majority"
    out["Momentum_agreement"] = _magree

    # ---- momentum vs price divergence --------------------------------------
    # The single most useful thing a momentum composite can tell you: price
    # making a new high while momentum does not (or the reverse). Because the
    # four components are highly correlated with each other, this comparison
    # against PRICE carries more information than their mutual agreement.
    _div_lb = int(cfg["rs_slope_lb"]) if "rs_slope_lb" in cfg else 21
    _px_chg = out["Close"].pct_change(_div_lb).to_numpy(dtype=float)
    _mom_chg = pd.Series(_mom).diff(_div_lb).to_numpy()
    _div = np.full(n, "", dtype=object)
    _okdv = (~np.isnan(_px_chg)) & (~np.isnan(_mom_chg)) & _okm
    _div[_okdv] = "None"
    _div[_okdv & (_px_chg > 0) & (_mom_chg < 0)] = "Bearish divergence"
    _div[_okdv & (_px_chg < 0) & (_mom_chg > 0)] = "Bullish divergence"
    out["Momentum_divergence"] = _div

    # ---- description -------------------------------------------------------
    # Band says WHAT, agreement says HOW RELIABLE, divergence and the
    # overbought/oversold guard say WHETHER TO TRUST IT. An extreme reading
    # with a divergence is the classic exhaustion setup, so it overrides the
    # plain band text rather than being appended to it.
    _e_hi = float(cfg["mom_extreme_rsi_hi"]); _e_lo = float(cfg["mom_extreme_rsi_lo"])
    _s_up = float(cfg["stoch_upper"]);        _s_lo = float(cfg["stoch_lower"])
    _mdesc = np.full(n, "", dtype=object)
    for i in range(n):
        if not _okm[i]:
            continue
        b = _mband[i]; ag = _magree[i]; dv = _div[i]
        r_hot = (not np.isnan(_ms_rsi[i])) and _ms_rsi[i] >= _e_hi
        r_cold = (not np.isnan(_ms_rsi[i])) and _ms_rsi[i] <= _e_lo
        k_hot = (not np.isnan(_k_arr[i])) and _k_arr[i] >= _s_up
        k_cold = (not np.isnan(_k_arr[i])) and _k_arr[i] <= _s_lo
        if r_hot and k_hot:
            _mdesc[i] = (f"{b} but overbought on RSI and Stochastic - stretched"
                         + (", bearish divergence" if dv == "Bearish divergence" else ""))
        elif r_cold and k_cold:
            _mdesc[i] = (f"{b} but oversold on RSI and Stochastic - stretched"
                         + (", bullish divergence" if dv == "Bullish divergence" else ""))
        elif dv in ("Bearish divergence", "Bullish divergence"):
            _mdesc[i] = f"{b} - {dv.lower()} vs price over {_div_lb} bars"
        elif ag in ("Bullish confirmed", "Bearish confirmed"):
            # "All components agree" overstates the case when the agreeing set
            # is RSI/ROC/Stoch, which are near-duplicates. MACD is the one
            # independent view, so name it explicitly when it is on board.
            if np.isnan(_ms_macd[i]):
                _mdesc[i] = f"{b} - return-based components agree, MACD unavailable"
            else:
                _mdesc[i] = (f"{b} - all {int(_m_navail[i])} components agree "
                             f"incl. MACD acceleration")
        elif ag in ("Bullish majority", "Bearish majority"):
            _dis = int(_mbear[i] if "Bull" in ag else _mbull[i])
            _macd_dissents = (not np.isnan(_ms_macd[i])) and (
                (_ms_macd[i] < 50) if "Bull" in ag else (_ms_macd[i] > 50))
            if _macd_dissents and _dis == 1:
                _mdesc[i] = (f"{b} - price momentum agrees but MACD shows "
                             f"deceleration")
            else:
                _mdesc[i] = f"{b} - majority agree, {_dis} dissenting"
        else:
            _mdesc[i] = f"{b} - components mixed, low conviction"
    out["Momentum_description"] = _mdesc

    out = out.copy()          # defragment after the momentum batch

    # ---- Keltner state ----
    _kcu = out["KC_upper"].to_numpy(dtype=float)
    _kcl = out["KC_lower"].to_numpy(dtype=float)
    _kcm = out["KC_mid"].to_numpy(dtype=float)
    kc_state = np.full(n, "", dtype=object)
    _kok = (~np.isnan(_kcu)) & (~np.isnan(_kcl)) & (~np.isnan(_c_np))
    kc_state[_kok & (_c_np >  _kcm)] = "Above mid"
    kc_state[_kok & (_c_np <= _kcm)] = "Below mid"
    kc_state[_kok & (_c_np >  _kcu)] = "Above upper"
    kc_state[_kok & (_c_np <  _kcl)] = "Below lower"
    out["KC_state"] = kc_state

    # ---- BB/KC squeeze: volatility compression ----
    # Recomputed at squeeze-specific config so changing bb_length elsewhere
    # does not silently redefine the squeeze.
    try:
        _sq_bb = work.ta.bbands(length=int(cfg["squeeze_bb_len"]),
                                std=float(cfg["squeeze_bb_std"]))
        # copy=True is REQUIRED: to_numpy() can hand back a read-only view of
        # the underlying block, and the warm-up assignment below writes in place.
        _sbu = _pick(_sq_bb, "BBU").to_numpy(dtype=float, copy=True)
        _sbl = _pick(_sq_bb, "BBL").to_numpy(dtype=float, copy=True)
        _sq_warm = max(int(cfg["squeeze_bb_len"]), _kc_warm)
        _sbu[idx < _sq_warm] = np.nan
        _sbl[idx < _sq_warm] = np.nan
    except Exception as _e:
        # Report rather than silently emitting an all-NaN squeeze column.
        print(f"  warn: BB/KC squeeze unavailable ({type(_e).__name__}: {_e})")
        _sbu = np.full(n, np.nan); _sbl = np.full(n, np.nan)

    # CONTINUOUS measure first -- a boolean discards compression DEPTH.
    # <1 = BB inside KC (squeeze); the lower the value, the tighter the coil.
    with np.errstate(invalid="ignore", divide="ignore"):
        _bb_w = _sbu - _sbl
        _kc_w = _kcu - _kcl
        _ratio = np.where((_kc_w > 0) & (~np.isnan(_bb_w)), _bb_w / _kc_w, np.nan)
    out["KC_squeeze_ratio"] = np.round(_ratio, 4)

    # Percentile of the ratio vs THIS stock's own history -> 0 = tightest coil
    # it has seen. Reported regardless of mode so the field is always usable.
    _sq_win = min(int(cfg["squeeze_pctl_win"]), n)
    _sq_ser = pd.Series(_ratio, index=out.index)
    out["KC_squeeze_pctl"] = np.round(
        _sq_ser.rolling(_sq_win, min_periods=max(30, _sq_win // 4))
               .rank(pct=True) * 100.0, 2)

    _mode = str(cfg["squeeze_mode"]).strip().lower()
    if _mode not in ("pctl", "ratio"):
        print(f"  config: unknown squeeze_mode '{_mode}' -> using 'pctl'")
        _mode = "pctl"

    sq = np.full(n, "", dtype=object)
    _sok = (~np.isnan(_ratio)) & _kok
    if _mode == "ratio":
        sq[_sok] = "No"
        sq[_sok & (_ratio < float(cfg["squeeze_ratio_thr"]))] = "Yes"
    else:
        _pv = out["KC_squeeze_pctl"].to_numpy(dtype=float)
        _pok = _sok & (~np.isnan(_pv))
        sq[_pok] = "No"
        sq[_pok & (_pv <= float(cfg["squeeze_pctl"]))] = "Yes"
    out["KC_squeeze"] = sq

    # bars the squeeze has persisted -> longer compression, larger expansion
    sq_bars = np.full(n, np.nan)
    _run = 0
    for i in range(n):
        if sq[i] == "":
            continue
        _run = _run + 1 if sq[i] == "Yes" else 0
        sq_bars[i] = _run
    out["KC_squeeze_bars"] = sq_bars

    # release: compression ended -> direction taken from close vs KC mid
    sq_rel = np.full(n, "", dtype=object)
    _sqp = np.concatenate([[""], sq[:-1]])
    for i in range(n):
        if sq[i] == "No" and _sqp[i] == "Yes":
            sq_rel[i] = "Release Up" if _c_np[i] > _kcm[i] else "Release Down"
    out["KC_squeeze_release"] = sq_rel

    # ================= COMPOSITE VOLATILITY SCORE ===========================
    # Blends four measures of how WIDE the market is trading right now:
    #   BB bandwidth  -- standard-deviation envelope around a moving average
    #   ATR           -- true range, so gaps widen it (BB's std does not)
    #   Donchian width-- the actual high/low range over N bars
    #   Keltner width -- ATR envelope around an EMA
    #
    # DIRECTIONLESS BY DESIGN. Trend_score and Momentum_score run bearish ->
    # bullish; this runs COMPRESSED -> EXPANDED. A score of 90 says the stock is
    # trading in an unusually wide range for itself, not that it is bullish.
    # Reading it as a directional signal is the main way this field gets misused.
    #
    # Every component is converted to a PERCENTILE against the stock's own
    # history before blending. Raw widths are not comparable across stocks (or
    # even across regimes for one stock), and percentiles fix both problems.

    _vw_win = min(int(cfg["vol_width_pctl_win"]), n)
    _vw_mp = max(30, _vw_win // 4)

    # ---- 1. Bollinger bandwidth percentile ---------------------------------
    # NOT reusing BB_bandwidth_pctl: that field is ranked over bb_squeeze_win
    # (126 by default) because it feeds the BB squeeze flag, while the other
    # three components here rank over vol_width_pctl_win (252). Blending
    # percentiles computed on different lookbacks is incoherent -- the same
    # volatility would map to different percentiles purely by window length --
    # so BB is re-ranked on the shared window.
    _bb_w = out["BB_bandwidth"] if "BB_bandwidth" in out.columns else pd.Series(np.nan, index=out.index)
    _vs_bb_s = (_bb_w.rolling(_vw_win, min_periods=_vw_mp).rank(pct=True) * 100.0)
    out["BB_width_pctl"] = _vs_bb_s.round(2)
    _vs_bb = _vs_bb_s.to_numpy(dtype=float)

    # ---- 2. ATR percentile: already computed -------------------------------
    # ATR is the only component built on TRUE range, so it is the one that
    # reacts to overnight gaps -- which matters on NSE results and policy days.
    _vs_atr = out["ATR_pctl"].to_numpy(dtype=float)

    # ---- 3. Donchian width percentile --------------------------------------
    # DC_width_pct is a raw percentage; rank it so it sits on the same scale as
    # the two fields above rather than dominating or vanishing by magnitude.
    _dc_w = out["DC_width_pct"] if "DC_width_pct" in out.columns else pd.Series(np.nan, index=out.index)
    _vs_dc = (_dc_w.rolling(_vw_win, min_periods=_vw_mp).rank(pct=True) * 100.0)
    out["DC_width_pctl"] = _vs_dc.round(2)
    _vs_dc = _vs_dc.to_numpy(dtype=float)

    # ---- 4. Keltner width percentile ---------------------------------------
    _kc_w = out["KC_width_pct"] if "KC_width_pct" in out.columns else pd.Series(np.nan, index=out.index)
    _vs_kc = (_kc_w.rolling(_vw_win, min_periods=_vw_mp).rank(pct=True) * 100.0)
    out["KC_width_pctl"] = _vs_kc.round(2)
    _vs_kc = _vs_kc.to_numpy(dtype=float)

    # No Vol_*_sub columns are emitted: each is byte-identical to the *_pctl
    # column just written above (BB_width_pctl, ATR_pctl, DC_width_pctl,
    # KC_width_pctl). Those are the canonical names; the blend uses the arrays.

    # ---- weighted blend, renormalized over available components ------------
    _VW = {"bb": float(cfg["vol_w_bb"]), "atr": float(cfg["vol_w_atr"]),
           "donchian": float(cfg["vol_w_donchian"]), "keltner": float(cfg["vol_w_keltner"])}
    _vol, _v_navail, _vb = _weighted_blend(
        n, {"bb": _vs_bb, "atr": _vs_atr, "donchian": _vs_dc, "keltner": _vs_kc},
        _VW, int(cfg["vol_min_components"]))
    out["Volatility_score"] = np.round(_vol, 2)
    out["Volatility_components_n"] = _v_navail

    # ---- effective independence -------------------------------------------
    # These four are two families, not four views. Keltner width IS ATR times a
    # multiplier (they correlate ~0.96), and BB bandwidth tracks Donchian width
    # closely (~0.93) because both widen with dispersion around a mean.
    #   family A: ATR + Keltner      -- true-range based, reacts to GAPS
    #   family B: Bollinger + Donchian -- close/extreme based, ignores gaps
    # This field reports how many distinct families back the reading, so a
    # "full" four-component score is not mistaken for four independent votes.
    _fam_tr = (~np.isnan(_vs_atr)) | (~np.isnan(_vs_kc))
    _fam_rng = (~np.isnan(_vs_bb)) | (~np.isnan(_vs_dc))
    out["Volatility_distinct_views"] = (_fam_tr.astype(int) + _fam_rng.astype(int))

    # Gap sensitivity: the true-range family minus the close-based family.
    # A large positive value means ATR/Keltner see far more volatility than
    # Bollinger/Donchian do -- the signature of OVERNIGHT GAP risk, which is
    # exactly what a close-based measure misses. On NSE this spikes around
    # results and policy announcements.
    # nanmean over an all-NaN warm-up column is a legitimate NaN, not an error,
    # so the "Mean of empty slice" warning is suppressed rather than printed
    # once per symbol.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        _tr_fam = np.nanmean(np.vstack([_vs_atr, _vs_kc]), axis=0)
        _rng_fam = np.nanmean(np.vstack([_vs_bb, _vs_dc]), axis=0)
    out["Volatility_gap_premium"] = np.round(_tr_fam - _rng_fam, 2)

    out["Volatility_score_basis"] = _vb

    # ---- readable band (percentile cut points, NOT directional) ------------
    _vvl = float(cfg["vol_band_very_low"]); _vl = float(cfg["vol_band_low"])
    _vh = float(cfg["vol_band_high"]);      _vvh = float(cfg["vol_band_very_high"])
    _vband = np.full(n, "", dtype=object)
    _okv = ~np.isnan(_vol)
    _vband[_okv & (_vol < _vvl)] = "Very Low"
    _vband[_okv & (_vol >= _vvl) & (_vol < _vl)] = "Low"
    _vband[_okv & (_vol >= _vl) & (_vol < _vh)] = "Normal"
    _vband[_okv & (_vol >= _vh) & (_vol < _vvh)] = "High"
    _vband[_okv & (_vol >= _vvh)] = "Very High"
    out["Volatility_band"] = _vband

    # ---- direction of change: expanding or contracting? --------------------
    # More actionable than the level. Volatility mean-reverts, so a LOW score
    # that is rising is the setup; a HIGH score that is rising is usually late.
    _v_lb = int(cfg["vol_expansion_lb"])
    _v_jump = float(cfg["vol_expansion_jump"])
    _vol_chg = pd.Series(_vol).diff(_v_lb).to_numpy()
    _vtrend = np.full(n, "", dtype=object)
    _okvc = (~np.isnan(_vol_chg)) & _okv
    _vtrend[_okvc] = "Stable"
    _vtrend[_okvc & (_vol_chg >= _v_jump)] = "Expanding"
    _vtrend[_okvc & (_vol_chg <= -_v_jump)] = "Contracting"
    out["Volatility_trend"] = _vtrend
    out["Volatility_change"] = np.round(_vol_chg, 2)

    # ---- squeeze consensus --------------------------------------------------
    # Two independent squeeze definitions already exist (BB bandwidth percentile
    # and the BB/KC ratio). Agreement between them is stronger evidence of real
    # compression than either alone, since they use different band maths.
    _sq_min = int(cfg["vol_squeeze_min"])
    _bb_sq = (out["BB_squeeze"].to_numpy(dtype=object) == "Yes")
    _kc_sq = (out["KC_squeeze"].to_numpy(dtype=object) == "Yes"
              if "KC_squeeze" in out.columns else np.zeros(n, dtype=bool))
    _sq_n = _bb_sq.astype(int) + _kc_sq.astype(int)
    out["Volatility_squeeze_n"] = _sq_n
    _sq_state = np.full(n, "", dtype=object)
    _oksq = _okv
    _sq_state[_oksq] = "No"
    _sq_state[_oksq & (_sq_n == 1)] = "Partial"
    _sq_state[_oksq & (_sq_n >= _sq_min)] = "Confirmed"
    out["Volatility_squeeze"] = _sq_state

    # Bars since a confirmed squeeze last ended. A squeeze and an expansion are
    # mutually exclusive AT THE SAME BAR -- by the time the score has risen
    # enough to read "Expanding", the bands have already widened and the squeeze
    # flag has switched off. So the release event is "expanding shortly AFTER a
    # confirmed squeeze", not "expanding during one".
    _sq_conf = (_sq_state == "Confirmed")
    _bars_since_sq = _bars_since(_sq_conf)
    out["Volatility_bars_since_squeeze"] = _bars_since_sq

    # ---- description --------------------------------------------------------
    # Level alone is nearly useless: what matters is level PLUS direction, and
    # whether a squeeze is resolving. A confirmed squeeze that is expanding is
    # the tradable event; a high score that is contracting is a move ending.
    _vdesc = np.full(n, "", dtype=object)
    _rel_win = int(cfg["vol_release_window"])
    for i in range(n):
        if not _okv[i]:
            continue
        b = _vband[i]; tr = _vtrend[i]; sq = _sq_state[i]
        _recent_sq = (not np.isnan(_bars_since_sq[i])) and _bars_since_sq[i] <= _rel_win
        if tr == "Expanding" and _recent_sq:
            _vdesc[i] = (f"Squeeze releasing - expanding {int(_bars_since_sq[i])} "
                         f"bars after compression")
        elif sq == "Confirmed":
            _vdesc[i] = f"{b} volatility, squeeze confirmed - expansion pending"
        elif tr == "Expanding" and b in ("High", "Very High"):
            _vdesc[i] = f"{b} volatility and still expanding - move may be extended"
        elif tr == "Expanding":
            _vdesc[i] = f"{b} volatility, expanding - range widening"
        elif tr == "Contracting" and b in ("Very Low", "Low"):
            _vdesc[i] = f"{b} volatility, contracting - coiling, watch for breakout"
        elif tr == "Contracting":
            _vdesc[i] = f"{b} volatility, contracting - move losing energy"
        else:
            _vdesc[i] = f"{b} volatility, stable"
    out["Volatility_description"] = _vdesc

    out = out.copy()          # defragment after the volatility batch

    # Defragment again after the Donchian/ROC/Keltner inserts.
    out = out.copy()

    # ================= RELATIVE STRENGTH vs BENCHMARK INDICES ===============
    # RS answers a different question from every other field here: not "is this
    # stock rising?" but "is it rising FASTER than the market?". A stock can be
    # in a clean uptrend on every other indicator and still be dead money if the
    # index is outrunning it -- that is exactly what RS exposes.
    #
    # Benchmark closes are joined on DATE, never positionally: an index series
    # with different holiday coverage would otherwise silently shift every value.
    BENCHES = resolve_bench_config(cfg)
    rs_labels_done = []
    if benchmarks:
        _rs_slope_lb  = int(cfg["rs_slope_lb"])
        _rs_pctl_win  = int(cfg["rs_pctl_win"])
        _rs_roc_p     = int(cfg["rs_roc_period"])

        _dates = pd.to_datetime(out["Date"])
        for _stem, _lab in BENCHES:
            _bdf = benchmarks.get(_lab)
            if _bdf is None or "Close" not in getattr(_bdf, "columns", []):
                continue
            # --- align benchmark to the stock's dates (left join, no fill) ---
            _b = (_bdf[["Date", "Close"]]
                  .assign(Date=lambda d: pd.to_datetime(d["Date"]))
                  .drop_duplicates(subset="Date")
                  .rename(columns={"Close": "_bclose"}))
            _m = pd.DataFrame({"Date": _dates}).merge(_b, on="Date", how="left")
            _bc = pd.to_numeric(_m["_bclose"], errors="coerce").replace(0, np.nan)
            # Forward-fill ONLY across gaps where the index did not trade but the
            # stock did. Leading NaNs stay NaN so RS never starts before the
            # benchmark history does.
            _bc = _bc.ffill()
            _bc.index = out.index

            _cov = float(_bc.notna().mean())
            if _cov < 0.5:
                print(f"  warn: benchmark {_lab} covers only {_cov:.0%} of the "
                      f"stock's dates -> RS_{_lab}_* left blank")
                continue

            # --- 1. Excess return: the readable headline --------------------
            # Stock % change minus benchmark % change over the same window, in
            # percentage points. Unlike the raw RS ratio (whose level depends on
            # both series' scales and means nothing on its own), this is directly
            # interpretable: +8 means the stock beat the index by 8 points.
            _sr = (out["Close"] / out["Close"].shift(_rs_roc_p) - 1.0) * 100.0
            _br = (_bc / _bc.shift(_rs_roc_p) - 1.0) * 100.0
            out[f"RS_{_lab}_excess_pct"] = _sr - _br

            # --- 2. RS line slope: is the outperformance building? -----------
            # Computed on the RS line (Close / benchmark) so it captures the
            # trend in relative performance, not just the latest window.
            _rs = out["Close"] / _bc
            out[f"RS_{_lab}_slope"] = (_rs / _rs.shift(_rs_slope_lb) - 1.0) * 100.0

            # --- 3. Percentile rank of the RS line ---------------------------
            # Self-calibrating per stock: 100 = strongest relative position in
            # its own recent history. This is the field to rank a universe on.
            _w = min(_rs_pctl_win, n)
            out[f"RS_{_lab}_pctl"] = (_rs.rolling(_w, min_periods=max(30, _w // 4))
                                        .rank(pct=True) * 100.0)
            rs_labels_done.append(_lab)

    # ---- headline RS score / state from the primary benchmark -------------
    if rs_labels_done:
        _prim = str(cfg["rs_primary"]).strip()
        if _prim not in rs_labels_done:
            print(f"  config: rs_primary '{_prim}' unavailable -> using {rs_labels_done[0]}")
            _prim = rs_labels_done[0]
        _ex = out[f"RS_{_prim}_excess_pct"].to_numpy(dtype=float)
        _sat = float(cfg["rs_saturate_pct"])
        with np.errstate(invalid="ignore"):
            out["RS_score"] = np.round(50.0 + 50.0 * np.clip(_ex / _sat, -1.0, 1.0), 2)

        _sl = out[f"RS_{_prim}_slope"].to_numpy(dtype=float)
        _st = np.full(n, "", dtype=object)
        _ok = ~np.isnan(_ex)
        _st[_ok & (_ex > 0)]  = "Outperforming"
        _st[_ok & (_ex <= 0)] = "Underperforming"
        _bo = _ok & (~np.isnan(_sl))
        _st[_bo & (_ex > 0)  & (_sl > 0)]  = "Outperforming, strengthening"
        _st[_bo & (_ex > 0)  & (_sl <= 0)] = "Outperforming, weakening"
        _st[_bo & (_ex <= 0) & (_sl < 0)]  = "Underperforming, weakening"
        _st[_bo & (_ex <= 0) & (_sl >= 0)] = "Underperforming, improving"
        out["RS_state"] = _st

        # ---- agreement across benchmarks --------------------------------
        # Leading the large-cap index but lagging the broad market (or the
        # reverse) says WHERE the strength is coming from, which a single
        # benchmark cannot show.
        if len(rs_labels_done) >= 2:
            _l0, _l1 = rs_labels_done[0], rs_labels_done[1]
            _a  = out[f"RS_{_l0}_excess_pct"].to_numpy(dtype=float)
            _b2 = out[f"RS_{_l1}_excess_pct"].to_numpy(dtype=float)
            _ag = np.full(n, "", dtype=object)
            _v = (~np.isnan(_a)) & (~np.isnan(_b2))
            _ag[_v & (_a > 0)  & (_b2 > 0)]  = "Leads both"
            _ag[_v & (_a <= 0) & (_b2 <= 0)] = "Lags both"
            _ag[_v & (_a > 0)  & (_b2 <= 0)] = f"Leads {_l0} only"
            _ag[_v & (_a <= 0) & (_b2 > 0)]  = f"Leads {_l1} only"
            out["RS_agreement"] = _ag

        out = out.copy()          # defragment after the RS batch

    # ---- index files: blank the volume-derived fields --------------------
    # Yahoo reports Volume as 0 for Indian indices. Every field below is then
    # either 0, NaN, or a ratio of zeros -- numbers that LOOK computed but mean
    # nothing. Blanking them keeps the schema identical to an equity TA file
    # while making it obvious the data was never there.
    if is_index:
        _vol_fields = ["Vol_SMA_fast", "Vol_SMA_slow", "Vol_ratio", "Vol_z",
                       "Vol_pctile", "Vol_spike", "Vol_state", "Vol_trend",
                       "Vol_price_confirm", "VWAP", "VWAP_roll", "px_vs_VWAP_pct",
                       "px_vs_VWAP_roll_pct", "VWMA", "OBV", "OBV_slope", "AD",
                       "ADOSC", "PVT", "CMF", "CMF_state", "MFI", "MFI_state",
                       "EFI", "EOM"]
        _zero_vol = bool((pd.to_numeric(out["Volume"], errors="coerce")
                          .fillna(0) == 0).mean() > 0.5)
        if _zero_vol:
            for _c in _vol_fields:
                if _c in out.columns:
                    # `dtype == object` misses pandas >= 3's native string
                    # dtype, which is what these text columns actually hold --
                    # that silently left them un-blanked. Test for numeric
                    # instead: anything non-numeric is a text/state column.
                    out[_c] = (np.nan if pd.api.types.is_numeric_dtype(out[_c])
                              else "")
            out = out.copy()

    # ================= FINAL DECISION (sequential gate) =====================
    # Runs LAST, after index volume-blanking, so an index file cannot pick up
    # volume conviction from fields that are about to be wiped.
    #
    # This is deliberately NOT an average of the five composite scores:
    #   * Volatility_score is a MAGNITUDE (compressed->expanded), not a
    #     direction. Averaging it into a directional score is a category error
    #     -- high volatility is neither bullish nor bearish.
    #   * Trend and Momentum share inputs (EMA slope and ROC both measure recent
    #     return), so averaging them double-counts the same price move.
    #   * An average destroys the reason: "62" tells you nothing about whether
    #     the setup failed on stage, on agreement, or on volume.
    #
    # Instead each group answers the question it can actually answer:
    #   Stage      -> may I participate at all?      (gate)
    #   Trend      -> which direction?               (direction)
    #   Momentum   -> is that direction tiring?      (veto only)
    #   Volume     -> is the move real?              (veto only)
    #   Volatility -> how big, and where is the stop? (sizing, never direction)
    _d_long  = float(cfg["decide_trend_long"])
    _d_short = float(cfg["decide_trend_short"])
    _d_conf  = float(cfg["decide_stage_conf"])
    _d_allow_short = bool(int(cfg["decide_allow_short"]))
    _d_req_vol = bool(int(cfg["decide_require_volume"]))
    _d_ext = float(cfg["decide_vol_extended"])
    _cap_stage = float(cfg["score_cap_stage_block"])
    _cap_trend = float(cfg["score_cap_trend_block"])
    _cap_mom   = float(cfg["score_cap_momentum_veto"])
    _cap_vol   = float(cfg["score_cap_volume_veto"])
    _cap_ext   = float(cfg["score_cap_vol_extended"])
    _sw_t = float(cfg["score_w_trend"]);   _sw_m = float(cfg["score_w_momentum"])
    _sw_v = float(cfg["score_w_volume"]);  _sw_s = float(cfg["score_w_stage"])

    def _col(name, default=""):
        if name in out.columns:
            return out[name].to_numpy(dtype=object)
        return np.full(n, default, dtype=object)

    def _num(name):
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").to_numpy(dtype=float)
        return np.full(n, np.nan)

    _stage_nm = _col("Stage_name")
    _stage_cf = _num("Stage_confidence")
    _stage_sub = _col("Stage_substage")
    _t_sc = _num("Trend_score")
    _t_ag = _col("Trend_agreement")
    _t_ds = _col("Trend_description")
    _m_sc = _num("Momentum_score")
    _m_dv = _col("Momentum_divergence")
    _m_ds = _col("Momentum_description")
    _v_pc = _col("Vol_price_confirm")
    _v_st = _col("Vol_state")
    _cmf_st = _col("CMF_state")
    _vol_sc = _num("Volatility_score")
    _vol_tr = _col("Volatility_trend")
    _vol_sq = _col("Volatility_squeeze")
    _vol_ds = _col("Volatility_description")
    _rs_ag = _col("RS_agreement")

    _decision = np.full(n, "", dtype=object)
    _conviction = np.full(n, np.nan)
    _blocker = np.full(n, "", dtype=object)
    _fdesc = np.full(n, "", dtype=object)
    _fscore = np.full(n, np.nan)

    # Hoisted out of the per-row loop below: neither closure captures loop
    # state directly -- both read `_raw`, which is (re)assigned before use on
    # every iteration -- so rebuilding them n times per symbol bought nothing.
    def _dist(v):
        return 0.0 if np.isnan(v) else (v - 50.0)

    def _capped(cap, lean_long):
        # a blocked bar keeps its directional LEAN but cannot exceed the
        # ceiling for the gate it failed. Longs are capped from above, shorts
        # (mirror) from below, so both sides stay rankable.
        if lean_long:
            return min(_raw, cap)
        return max(_raw, 100.0 - cap)

    for i in range(n):
        if np.isnan(_t_sc[i]) or not _stage_nm[i]:
            continue                      # not enough history yet

        # Raw directional strength, 0-100, BEFORE any gate ceiling. Built only
        # from the components that carry direction (Trend, Momentum) plus
        # quality weights (Volume, Stage). Volatility is intentionally absent --
        # it is magnitude, not direction. This is the "fill" that the gate then
        # caps. Each piece is expressed as distance from 50 so that a bearish
        # component pushes the score below 50 symmetrically.
        _raw = 50.0
        _raw += _sw_t * _dist(_t_sc[i])
        _raw += _sw_m * _dist(_m_sc[i] if not np.isnan(_m_sc[i]) else 50.0)
        # volume quality: signed by its up/down confirmation
        _vq = str(_v_pc[i])
        _vqd = (12.0 if _vq == "Confirmed Up" else -12.0 if _vq == "Confirmed Down"
                else 4.0 if _vq == "Weak Up" else -4.0 if _vq == "Weak Down" else 0.0)
        _raw += _sw_v * _vqd
        # stage confidence nudges toward the stage's natural direction
        _sc_dir = (1.0 if str(_stage_nm[i]).startswith("2")
                   else -1.0 if str(_stage_nm[i]).startswith("4") else 0.0)
        _raw += _sw_s * _sc_dir * (_dist(_stage_cf[i]) if not np.isnan(_stage_cf[i]) else 0.0)
        _raw = float(np.clip(_raw, 0.0, 100.0))

        gates = 0
        notes = []

        # --- GATE 1: Stage. Most losses come from trading the wrong stage,
        # so this vetoes everything below it regardless of indicator strength.
        st = str(_stage_nm[i])
        stage_ok_long = st.startswith("2")
        stage_ok_short = st.startswith("4") and _d_allow_short
        weak_conf = (not np.isnan(_stage_cf[i])) and _stage_cf[i] < _d_conf
        if weak_conf:
            # "Possibly stage 2" is not stage 2. Treat low confidence as unclear.
            _decision[i] = "STAND ASIDE"
            _blocker[i] = "Stage unclear"
            _fdesc[i] = (f"{st} but stage confidence only {_stage_cf[i]:.0f} "
                         f"- regime not established")
            _conviction[i] = 0
            _fscore[i] = 50.0            # pinned neutral: no participation
            continue
        if not (stage_ok_long or stage_ok_short):
            _decision[i] = "STAND ASIDE"
            _blocker[i] = "Stage"
            _reason = ("no new positions in a basing/topping stage"
                       if st.startswith(("1", "3"))
                       else "declining stage, shorts disabled")
            _fdesc[i] = f"{st} - {_reason}"
            _conviction[i] = 0
            _fscore[i] = round(min(_raw, _cap_stage), 2)
            continue
        gates += 1
        side = "LONG" if stage_ok_long else "SHORT"

        # --- GATE 2: Trend direction, and whether it is internally coherent.
        want_long = side == "LONG"
        dir_ok = (_t_sc[i] >= _d_long) if want_long else (_t_sc[i] <= _d_short)
        mixed = str(_t_ag[i]) == "Mixed"
        ranging = "range" in str(_t_ds[i]).lower()
        if not dir_ok:
            _decision[i] = "WAIT"
            _blocker[i] = "Trend"
            _fdesc[i] = (f"{st}, but trend score {_t_sc[i]:.0f} does not confirm "
                         f"{side.lower()} - no directional edge")
            _conviction[i] = 1
            _fscore[i] = round(_capped(_cap_trend, want_long), 2)
            continue
        if mixed or ranging:
            # A mid-range score built from disagreeing components is ambiguous,
            # not mildly positive. Ambiguity means no position.
            _decision[i] = "WAIT"
            _blocker[i] = "Trend agreement" if mixed else "Weak ADX"
            _why = ("components disagree" if mixed
                    else "ADX shows no trend, this is a range")
            _fdesc[i] = f"{st}, trend {_t_sc[i]:.0f} but {_why} - wait for alignment"
            _conviction[i] = 1
            _fscore[i] = round(_capped(_cap_trend, want_long), 2)
            continue
        gates += 1

        # --- GATE 3: Momentum. Can only subtract. Agreement here adds little
        # because Momentum shares inputs with Trend.
        stretched = "stretched" in str(_m_ds[i]).lower()
        div = str(_m_dv[i])
        against = ((div == "Bearish divergence" and want_long) or
                   (div == "Bullish divergence" and not want_long))
        if stretched:
            _decision[i] = "NO NEW ENTRY"
            _blocker[i] = "Momentum stretched"
            _fdesc[i] = (f"{st}, trend {_t_sc[i]:.0f} {side.lower()} but momentum "
                         f"overextended - hold existing, do not add")
            _conviction[i] = 2
            _fscore[i] = round(_capped(_cap_mom, want_long), 2)
            continue
        if against:
            _decision[i] = "HOLD / TIGHTEN STOP"
            _blocker[i] = "Momentum divergence"
            _fdesc[i] = (f"{st}, trend {_t_sc[i]:.0f} {side.lower()} but momentum "
                         f"diverging - trend tiring, tighten stop")
            _conviction[i] = 2
            _fscore[i] = round(_capped(_cap_mom, want_long), 2)
            continue
        gates += 1

        # --- GATE 4: Volume. Confirms the move is real; never sets direction.
        vconf = str(_v_pc[i])
        vol_weak = vconf.startswith("Weak") or str(_v_st[i]) == "Dry-up"
        cmf_against = ((str(_cmf_st[i]) == "Distribution" and want_long) or
                       (str(_cmf_st[i]) == "Accumulation" and not want_long))
        if _d_req_vol and (vol_weak or cmf_against):
            _decision[i] = "ENTER SMALL"
            _blocker[i] = "Volume"
            _why = "participation is thin" if vol_weak else "money flow disagrees"
            _fdesc[i] = (f"{st}, trend {_t_sc[i]:.0f} {side.lower()} confirmed but "
                         f"{_why} - reduce size, breakouts without volume fail more often")
            _conviction[i] = 3
            _fscore[i] = round(_capped(_cap_vol, want_long), 2)
            continue
        gates += 1

        # --- GATE 5: Volatility. Sizing and timing only, never direction.
        extended = (not np.isnan(_vol_sc[i])) and _vol_sc[i] >= _d_ext \
                   and str(_vol_tr[i]) == "Expanding"
        releasing = str(_vol_sq[i]) == "Confirmed" or (
            "squeeze releasing" in str(_vol_ds[i]).lower())
        if extended:
            _decision[i] = "ENTER SMALL"
            _blocker[i] = "Volatility extended"
            _fdesc[i] = (f"{st}, {side.lower()} setup confirmed but volatility "
                         f"{_vol_sc[i]:.0f} and still expanding - late, size down "
                         f"and widen stop")
            _conviction[i] = 4
            _fscore[i] = round(_capped(_cap_ext, want_long), 2)
            continue
        gates += 1

        # --- all gates passed --------------------------------------------
        _conviction[i] = gates
        # Full score: no ceiling. A releasing squeeze and benchmark leadership
        # are the two things that push a clean setup toward the extremes.
        _full = _raw
        lead = "Leads both" in str(_rs_ag[i])
        if lead:
            _full += 4.0 if want_long else -4.0
        if releasing:
            _full += 3.0 if want_long else -3.0
        _fscore[i] = round(float(np.clip(_full, 0.0, 100.0)), 2)
        extra = []
        if releasing:
            extra.append("volatility releasing from compression")
        if lead:
            extra.append("outperforming both benchmarks")
        if str(_stage_sub[i]) == "early":
            extra.append("early in the stage")
        elif str(_stage_sub[i]) == "late":
            extra.append("late in the stage, reduce size")
        tail = ("; " + ", ".join(extra)) if extra else ""
        _decision[i] = f"ENTER {side}"
        _blocker[i] = "None"
        _fdesc[i] = (f"{st}, trend {_t_sc[i]:.0f} with {str(_t_ag[i]).lower()}, "
                     f"momentum and volume confirm{tail}")

    out["Final_decision"] = _decision
    out["Final_score"] = _fscore
    out["Final_conviction"] = _conviction
    out["Final_blocker"] = _blocker
    out["Final_description"] = _fdesc

    # ---- what did the call look like 5 / 10 trading sessions ago? ----------
    # A snapshot, not a recomputation: shifting the already-decided description
    # lets a reader compare today's read against the read from 5/10 sessions
    # ago (e.g. "was ENTER LONG then, now STAND ASIDE") without opening an
    # older TA file. Blank wherever no description existed that far back yet
    # (warm-up, or the symbol's whole history is shorter than the lookback).
    out["Final_description_5d_ago"] = pd.Series(_fdesc, index=out.index).shift(5).fillna("")
    out["Final_description_10d_ago"] = pd.Series(_fdesc, index=out.index).shift(10).fillna("")

    # Index files are benchmarks, not tradables. Their volume fields were just
    # blanked, so the volume gate cannot evaluate and any ENTER signal would be
    # meaningless. Blank the whole decision rather than emit a call nobody can act on.
    if is_index:
        for _c in ("Final_decision", "Final_blocker", "Final_description"):
            out[_c] = ""
        for _c in ("Final_score", "Final_conviction"):
            out[_c] = np.nan

    out = out.copy()

    # round indicator columns for a tidy file
    ind_cols = ["RSI_14", "RSI_upper", "RSI_lower", "RSI_z",
                "MACD", "MACD_signal", "MACD_hist", "PPO", "MACD_hist_z"]
    ind_cols += [f"EMA_{p}" for p in EMA_PERIODS]
    ind_cols += [f"px_vs_EMA_{p}" for p in EMA_PERIODS]
    ind_cols += [f"EMA_{p}_slope" for p in EMA_PERIODS]
    ind_cols += ["ST_line", "ST_dist_pct", "ST_bars_since_flip",
                 "MS_swing_high", "MS_swing_low",
                 "MS_last_HH", "MS_last_HL", "MS_last_LH", "MS_last_LL",
                 "EMA_location_score", "EMA_structure_score", "EMA_slope_score", "EMA_score",
                 "BB_upper", "BB_mid", "BB_lower",
                 "ADX_14", "ATR_14", "Stoch_K", "Stoch_D"]
    ind_cols += ['ATR_pct', 'ATR_pctl', 'Stop_long', 'Stop_short']
    ind_cols += ['DI_plus', 'DI_minus', 'BB_pctB', 'BB_bandwidth', 'BB_bandwidth_pctl']
    ind_cols += ['Stage', 'Stage_MA', 'Stage_MA_slope', 'Stage_days', 'Prev_stage', 'Prev_stage_days', 'Stage_start_price', 'Stage_return_pct', 'Stage_maturity_pct', 'Stage_confidence']
    ind_cols += ['Vol_SMA_fast', 'Vol_SMA_slow', 'Vol_ratio', 'Vol_z', 'Vol_pctile', 'VWAP_roll', 'px_vs_VWAP_pct', 'px_vs_VWAP_roll_pct', 'VWMA', 'OBV', 'OBV_slope', 'AD', 'ADOSC', 'PVT', 'CMF', 'MFI', 'EFI', 'EOM']
    ind_cols += ['DC_upper', 'DC_mid', 'DC_lower', 'DC_width_pct', 'DC_pct_rank',
                 'DC_bars_since_breakout',
                 'KC_upper', 'KC_mid', 'KC_lower', 'KC_width_pct', 'KC_pct_rank',
                 'KC_squeeze_ratio', 'KC_squeeze_pctl', 'KC_squeeze_bars', 'ROC_score']
    ind_cols += [f"ROC_{p}" for p in ROC_PERIODS]
    ind_cols += [f"ROC_{p}_z" for p in ROC_PERIODS]
    ind_cols += ["Trend_supertrend_sub", "Trend_adx_sub",
                 "Trend_structure_sub", "Trend_score"]
    ind_cols += ["Mom_macd_sub", "Momentum_score"]
    ind_cols += ["Final_score"]
    ind_cols += ["BB_width_pctl", "DC_width_pctl", "KC_width_pctl",
                 "Volatility_score", "Volatility_change",
                 "Volatility_bars_since_squeeze", "Volatility_gap_premium"]
    for _lab in rs_labels_done:
        ind_cols += [f"RS_{_lab}_excess_pct", f"RS_{_lab}_slope", f"RS_{_lab}_pctl"]
    if rs_labels_done:
        ind_cols += ["RS_score"]
    for c in ind_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(4)

    # --- output layout: related fields stay together in every generated CSV ---
    # Keep the groups explicit.  It makes the exported file easy to scan and
    # ensures newly added fields have an obvious, single place to be registered.
    field_groups = {
        # Source market data
        "price": ["Date", "Open", "High", "Low", "Close", "Volume"],
        # Moving averages and price location relative to them
        "moving_averages": (
            [f"EMA_{p}" for p in EMA_PERIODS]
            + [f"Below_EMA_{max(EMA_PERIODS)}"]
            + [f"px_vs_EMA_{p}" for p in EMA_PERIODS]
            + [f"EMA_{p}_slope" for p in EMA_PERIODS]
            + ["EMA_location_score", "EMA_structure_score", "EMA_slope_score",
               "EMA_score", "EMA_trend", "EMA_ribbon_n", "EMA_score_basis"]
        ),
        # Trend-following indicators and swing-based market structure
        "trend_and_structure": [
            "ST_line", "ST_dir", "ST_state", "ST_signal", "ST_dist_pct",
            "ST_bars_since_flip",
            "MS_structure", "MS_bias", "MS_event",
            "MS_swing_high", "MS_swing_low",
            "MS_last_HH", "MS_last_HL", "MS_last_LH", "MS_last_LL",
        ],
        # Composite trend score: EMA + Supertrend + ADX + market structure.
        # Sub-scores come first so a surprising Trend_score can be decomposed
        # in the CSV without recomputing anything.
        "trend_composite": [
            "Trend_supertrend_sub", "Trend_adx_sub",
            "Trend_structure_sub",
            "Trend_score", "Trend_band", "Trend_agreement",
            "Trend_bull_components", "Trend_bear_components",
            "Trend_components_n", "Trend_score_basis", "Trend_description",
        ],
        # Relative strength versus the benchmark indices
        "relative_strength": (
            [c for _lab in rs_labels_done for c in
             (f"RS_{_lab}_excess_pct", f"RS_{_lab}_slope", f"RS_{_lab}_pctl")]
            + ["RS_score", "RS_state", "RS_agreement"]
        ),
        # Momentum oscillators
        "momentum": [
            "RSI_14", "RSI_upper", "RSI_lower", "RSI_z", "RSI_state", "RSI_basis",
            "MACD", "MACD_signal", "MACD_hist", "PPO", "MACD_hist_z", "MACD_state",
            "Stoch_K", "Stoch_D", "Stoch_state", "Stoch_cross",
        ]
        + [f"ROC_{p}" for p in ROC_PERIODS]
        + [f"ROC_{p}_z" for p in ROC_PERIODS]
        + [f"ROC_{p}_z_basis" for p in ROC_PERIODS]
        + ["ROC_score", "ROC_state"],
        # Composite momentum score: RSI + ROC + MACD + Stochastic.
        # Sub-scores first so a surprising Momentum_score can be decomposed
        # directly in the CSV without recomputing anything.
        "momentum_composite": [
            "Mom_macd_sub", "Momentum_score", "Momentum_band", "Momentum_agreement",
            "Momentum_divergence",
            "Momentum_bull_components", "Momentum_bear_components",
            "Momentum_distinct_views",
            "Momentum_score_basis",
            "Momentum_description",
        ],
        # Bollinger, ADX, and ATR measures of volatility and trend strength
        "volatility_and_strength": [
            "BB_upper", "BB_mid", "BB_lower", "BB_pctB", "BB_bandwidth",
            "BB_bandwidth_pctl", "BB_squeeze", "BB_state",
            "ADX_14", "DI_plus", "DI_minus", "ADX_regime", "ADX_direction", "ADX_rising",
            "ATR_14", "ATR_pct", "ATR_pctl", "ATR_regime", "ATR_rising",
            "DC_upper", "DC_mid", "DC_lower", "DC_width_pct", "DC_pct_rank",
            "DC_state", "DC_bars_since_breakout",
            "KC_upper", "KC_mid", "KC_lower", "KC_width_pct", "KC_pct_rank",
            "KC_state", "KC_squeeze_ratio", "KC_squeeze_pctl", "KC_squeeze",
            "KC_squeeze_bars", "KC_squeeze_release",
        ],
        # Composite volatility score: BB + ATR + Donchian + Keltner.
        # DIRECTIONLESS: 0 = compressed, 100 = expanded. Not a bullish signal.
        "volatility_composite": [
            "BB_width_pctl", "DC_width_pctl", "KC_width_pctl",
            "Volatility_score", "Volatility_band", "Volatility_trend",
            "Volatility_change", "Volatility_squeeze", "Volatility_squeeze_n",
            "Volatility_bars_since_squeeze",
            "Volatility_distinct_views", "Volatility_gap_premium",
            "Volatility_components_n", "Volatility_score_basis",
            "Volatility_description",
        ],
        # Position-risk values and the combined trading read
        "risk_and_setup": [
            "Stop_long", "Stop_short", "Setup", "ATR_note",
        ],
        # Price/volume averages and relative volume
        "volume": [
            "VWAP", "VWAP_roll", "px_vs_VWAP_pct", "px_vs_VWAP_roll_pct", "VWMA",
            "Vol_SMA_fast", "Vol_SMA_slow", "Vol_ratio", "Vol_z", "Vol_pctile",
            "Vol_spike", "Vol_state", "Vol_trend", "Vol_price_confirm",
        ],
        # Accumulation/distribution and volume-derived oscillators
        "money_flow": [
            "OBV", "OBV_slope", "AD", "ADOSC", "PVT",
            "CMF", "CMF_state", "MFI", "MFI_state", "EFI", "EOM",
        ],
        # Weinstein market-cycle state and duration fields
        "stage": [
            "Stage", "Stage_name", "Stage_substage", "Stage_confidence", "Stage_days",
            "Stage_maturity_pct", "Stage_start_date",
            "Stage_start_price", "Stage_return_pct", "Stage_transition",
            "Prev_stage", "Prev_stage_name", "Prev_stage_days", "Stage_MA", "Stage_MA_slope",
        ],
    }
    desired = [column for group in field_groups.values() for column in group]
    ordered = [column for column in desired if column in out.columns]
    ordered_set = set(ordered)
    ordered += [column for column in out.columns if column not in ordered_set]

    # Drop hidden columns LAST, after ordering, so the layout is unchanged for
    # whatever remains. Date and OHLCV are never hidden -- without them the file
    # is not a time series and cannot be rejoined to anything.
    _protected = {"Date", "Open", "High", "Low", "Close", "Volume"}
    _hidden = resolve_hidden_fields(cfg, ordered) - _protected
    if _hidden:
        ordered = [c for c in ordered if c not in _hidden]

    out = out[ordered]

    return out


# Tracks whether the hidden-fields message has been printed, so a multi-symbol
# run reports it once rather than once per symbol.
_HIDE_REPORTED = set()
_HIDE_SEEN = set()      # hidden_fields names that matched a column at least once
_HIDE_PENDING = set()   # names not yet matched; reported at end of run if still unmatched


def resolve_hidden_fields(cfg, available):
    """Return the set of columns to drop from the written file.

    `available` is the actual column list, so a name in hidden_fields that does
    not match anything is reported as a probable typo instead of silently doing
    nothing. Fields are only hidden from OUTPUT -- they are computed either way,
    so every score and description is unaffected by what is listed here."""
    if int(cfg.get("show_all_fields", 0)):
        if not _HIDE_REPORTED:
            print("  fields: show_all_fields=1 -> hidden_fields ignored, emitting all columns")
            _HIDE_REPORTED.add("shown")
        return set()

    raw = str(cfg.get("hidden_fields", "") or "")
    names = [t.strip() for t in raw.replace("\n", ",").replace("\t", ",").split(",")]
    names = [t for t in names if t]
    if not names:
        return set()

    avail = set(available)
    hide = {t for t in names if t in avail}
    # Track names that have matched at least once anywhere in this run. A field
    # can be legitimately absent for one symbol but present for another --
    # RS_NIFTY50_* exists for equities but never for index rows -- so warning
    # per symbol would flag correct config as a typo. Only names that never
    # resolve for ANY symbol are worth reporting.
    _HIDE_SEEN.update(hide)
    unknown = sorted({t for t in names if t not in _HIDE_SEEN})
    if unknown and not _HIDE_REPORTED:
        _HIDE_PENDING.clear()
        _HIDE_PENDING.update(unknown)
    return hide


def report_unknown_hidden_fields():
    """Report hidden_fields names that never matched a column in this run.

    Called once after all symbols are processed, not per symbol."""
    never = sorted(_HIDE_PENDING - _HIDE_SEEN)
    if never and not _HIDE_REPORTED:
        print(f"  fields: {len(never)} name(s) in hidden_fields never matched a column "
              f"(check for typos): {', '.join(never[:8])}"
              + (" ..." if len(never) > 8 else ""))
        _HIDE_REPORTED.add("warned")


def normalize_symbols_frame(symbols):
    """Clean symbols.csv the same way bulk_load.py does.

    A trailing space in the CSV ("NIFTY50 ,1,index ") is invisible in a text
    editor but produces filenames like "NIFTY50 .csv" and a kind of "index ",
    so the benchmark lookup silently misses and RS_* comes out blank. Both
    scripts must strip identically or they disagree about what a symbol is
    called. Also coerces `active` to a real integer, since a quoted or
    space-padded value makes `active == 1` false for every row."""
    symbols = symbols.copy()
    symbols.columns = [str(c).strip() for c in symbols.columns]
    for col in symbols.columns:
        # Strip every non-numeric column. Checking dtype via select_dtypes is
        # deprecated in pandas 3 for str vs object, so test the values instead.
        if not pd.api.types.is_numeric_dtype(symbols[col]):
            symbols[col] = symbols[col].astype(str).str.strip()

    if "active" in symbols.columns:
        symbols["active"] = pd.to_numeric(symbols["active"], errors="coerce").fillna(0).astype(int)

    if "kind" not in symbols.columns:
        symbols["kind"] = "equity"
    symbols["kind"] = (symbols["kind"].astype(str).str.strip().str.lower()
                       .replace({"": "equity", "nan": "equity", "none": "equity"}))

    known = {"equity", "index"}
    bad = sorted(set(symbols["kind"]) - known)
    if bad:
        print(f"  symbols: unrecognised kind value(s) {bad} -> treated as equity")
        symbols.loc[~symbols["kind"].isin(known), "kind"] = "equity"
    return symbols


def load_benchmarks(raw_dir: Path, cfg):
    """Read the benchmark index CSVs once, for reuse across every symbol.

    Returns {label: DataFrame}. A missing file is reported and skipped rather
    than aborting the run -- RS fields are simply absent for that benchmark."""
    out = {}
    for stem, label in resolve_bench_config(cfg):
        path = raw_dir / f"{stem}.csv"
        if not path.exists():
            alt = [p.name for p in raw_dir.glob("*.csv")
                   if p.stem.strip() == stem and p.stem != stem]
            if alt:
                print(f"  benchmark: {path.name} not found, but {alt[0]!r} exists "
                      f"-> remove the space from symbols.csv and re-run bulk_load")
            else:
                print(f"  benchmark: {path.name} not found -> RS_{label}_* will be skipped")
            continue
        try:
            b = pd.read_csv(path)
            b.columns = [str(c).strip() for c in b.columns]
            if "Date" not in b.columns or "Close" not in b.columns:
                print(f"  benchmark: {path.name} needs Date and Close columns -> skipped")
                continue
            b["Date"] = pd.to_datetime(b["Date"])
            b = (b.drop_duplicates(subset="Date")
                   .sort_values("Date").reset_index(drop=True))
            out[label] = b
            print(f"  benchmark: {label} loaded ({len(b)} rows, "
                  f"{b['Date'].min():%Y-%m-%d} to {b['Date'].max():%Y-%m-%d})")
        except Exception as e:
            print(f"  benchmark: failed to read {path.name} ({type(e).__name__}: {e})")
    return out


def process_symbol(sym, raw_dir: Path, out_dir: Path, cfg, benchmarks=None,
                   is_index=False):
    """Read raw_dir/SYMBOL.csv, compute indicators, write out_dir/SYMBOL_TA.csv."""
    sym = str(sym).strip()          # matches bulk_load.py's filename convention
    raw_path = raw_dir / f"{sym}.csv"
    if not raw_path.exists():
        # A stray space in symbols.csv produces "NIFTY50 .csv" on disk; name it
        # explicitly rather than leaving the user to spot an invisible character.
        alt = [p.name for p in raw_dir.glob("*.csv")
               if p.stem.strip() == sym and p.stem != sym]
        if alt:
            return (f"skip {sym}: raw file not found ({raw_path.name}); "
                    f"found {alt[0]!r} -- remove the space from the filename")
        return f"skip {sym}: raw file not found ({raw_path.name})"

    df = pd.read_csv(raw_path)
    if "Date" not in df.columns:
        return f"skip {sym}: no Date column"

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)

    min_rows = int(cfg["min_rows"])
    if len(df) < min_rows:
        return f"skip {sym}: <{min_rows} rows (only {len(df)})"

    result = add_indicators(df, cfg, benchmarks=benchmarks, is_index=is_index)
    out_path = out_dir / f"{sym}_TA.csv"
    result.to_csv(out_path, index=False)
    return f"ok   {sym}: {len(result)} rows -> {out_path.name}"


def main():
    # Reset run-scoped state so a second main() call in the same process (e.g.
    # a test importing this module) reports hidden-field typos fresh rather
    # than inheriting "already reported" from a prior run.
    _HIDE_REPORTED.clear()
    _HIDE_SEEN.clear()
    _HIDE_PENDING.clear()

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE),
                    help="market_data base folder (contains config/, raw/, ta/)")
    ap.add_argument("--all", action="store_true",
                    help="process every symbol in symbols.csv, ignoring the active flag")
    args = ap.parse_args()

    base    = Path(args.base)
    cfg_dir = base / "config"
    sym_csv = cfg_dir / "symbols.csv"
    cfg_csv = cfg_dir / "config.csv"
    raw_dir = base / "raw" / "equity"
    out_dir = base / "ta" / "equity"

    if not sym_csv.exists():
        sys.exit(f"Symbol list not found: {sym_csv}")
    if not raw_dir.exists():
        sys.exit(f"Raw data folder not found: {raw_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = pd.read_csv(sym_csv)
    if "symbol" not in symbols.columns:
        sys.exit(f"'symbol' column missing in {sym_csv}")

    # Strip whitespace and coerce types BEFORE the active filter -- a padded
    # "1 " would otherwise fail `active == 1` and silently drop every row.
    symbols = normalize_symbols_frame(symbols)

    # honour the active flag (same convention as bulk_load.py) unless --all
    if not args.all and "active" in symbols.columns:
        symbols = symbols[symbols.active == 1]

    # A 'kind' column (equity | index) lets benchmark rows live in the same
    # symbols.csv without being treated as tradable stocks. Index rows are
    # still processed so they get their own TA file, but RS is not computed
    # for them -- a benchmark's relative strength against itself is a constant.
    kinds = dict(zip(symbols["symbol"].astype(str), symbols["kind"].astype(str)))

    sym_list = symbols["symbol"].astype(str).tolist()
    if not sym_list:
        sys.exit("No symbols to process (check the active flag or use --all).")

    cfg = load_config(cfg_csv)
    bench = load_benchmarks(raw_dir, cfg)

    print(f"\nProcessing {len(sym_list)} symbols")
    print(f"  raw in : {raw_dir}")
    print(f"  TA out : {out_dir}\n")

    ok = 0
    for sym in sym_list:
        # index rows get every other indicator but no RS (self-comparison)
        _is_idx = kinds.get(sym) == "index"
        _bm = None if _is_idx else bench
        try:
            msg = process_symbol(sym, raw_dir, out_dir, cfg,
                                 benchmarks=_bm, is_index=_is_idx)
        except Exception as e:
            # Include the failing line so a bad symbol is diagnosable from the log
            # rather than just "ERROR SYM: <message>".
            tb = traceback.extract_tb(sys.exc_info()[2])
            where = ""
            for fr in reversed(tb):
                if Path(fr.filename).name == Path(__file__).name:
                    where = f" [line {fr.lineno}: {fr.line}]"
                    break
            msg = f"ERROR {sym}: {type(e).__name__}: {e}{where}"
        if msg.startswith("ok"):
            ok += 1
        print(msg)

    report_unknown_hidden_fields()
    print(f"\nDone: {ok}/{len(sym_list)} TA files written to {out_dir}")


if __name__ == "__main__":
    main()
