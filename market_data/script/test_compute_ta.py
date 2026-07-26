#!/usr/bin/env python3
"""Lightweight smoke test for compute_ta.py.

Not a full test suite (see the review notes on why one is needed): this is a
minimal safety net that runs add_indicators() on synthetic OHLCV data and
checks a handful of invariants that a broken edit would likely violate --
warm-up rows staying blank, no exceptions, composite scores landing in range,
and the two bugs fixed in this pass staying fixed. Run directly:

    python test_compute_ta.py
"""
import numpy as np
import pandas as pd

from compute_ta import DEFAULTS, add_indicators, resolve_hidden_fields


def make_synthetic_ohlcv(n=600, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    ret = rng.normal(0.0004, 0.015, n)
    close = 100.0 * np.cumprod(1.0 + ret)
    high = close * (1.0 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.006, n)))
    open_ = low + (high - low) * rng.random(n)
    volume = rng.integers(100_000, 5_000_000, n).astype(float)
    return pd.DataFrame({
        "Date": dates, "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    })


def main():
    cfg = dict(DEFAULTS)
    cfg["show_all_fields"] = 1  # keep hidden_fields columns so we can inspect them
    df = make_synthetic_ohlcv()
    out = add_indicators(df, cfg=cfg)
    n = len(out)

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # --- warm-up masking: ADX/ATR/Stoch must be NaN before they mature, and
    # (this is the bug fixed in this pass) any regime/state derived from them
    # must ALSO be blank over that same window, not just the raw indicator. ---
    adx_len = int(cfg["adx_length"])
    check(out["ADX_14"].iloc[:adx_len * 2 - 2].isna().all(),
          "ADX_14 not blank during warm-up")
    check((out["ADX_regime"].iloc[:adx_len * 2 - 2] == "").all(),
          "ADX_regime not blank during ADX warm-up (mask-ordering bug)")

    atr_len = int(cfg["atr_length"])
    check(out["ATR_14"].iloc[:atr_len - 1].isna().all(),
          "ATR_14 not blank during warm-up")
    check(out["ATR_pct"].iloc[:atr_len - 1].isna().all(),
          "ATR_pct not blank during ATR warm-up (mask-ordering bug)")
    check((out["ATR_regime"].iloc[:atr_len - 1] == "").all(),
          "ATR_regime not blank during ATR warm-up (mask-ordering bug)")

    sk, sd = int(cfg["stoch_k"]), int(cfg["stoch_d"])
    stoch_warm = sk + sd - 2
    check(out["Stoch_K"].iloc[:stoch_warm].isna().all(),
          "Stoch_K not blank during warm-up")
    check((out["Stoch_state"].iloc[:stoch_warm] == "").all(),
          "Stoch_state not blank during Stoch warm-up (mask-ordering bug)")

    # --- composite scores: bounded 0-100 wherever present ---
    for col in ("EMA_score", "Trend_score", "Momentum_score", "Volatility_score"):
        vals = out[col].dropna()
        check(len(vals) > 0, f"{col} is entirely blank")
        check(vals.between(0, 100).all(), f"{col} has values outside [0, 100]")

    # --- EMA_score min-components gate: basis must never be 'partial' with
    # fewer than ema_min_components sub-scores actually backing it ---
    check(not ((out["EMA_score"].notna()) & (out["EMA_ribbon_n"] == 0)).any(),
          "EMA_score present with zero ribbon components")

    # --- Final_decision: only emitted once Trend_score/Stage exist ---
    has_decision = out["Final_decision"].astype(str).str.len() > 0
    check((out.loc[has_decision, "Trend_score"].notna()).all(),
          "Final_decision emitted on a row without Trend_score")
    if has_decision.any():
        check(out.loc[has_decision, "Final_score"].between(0, 100).all(),
              "Final_score outside [0, 100] on a decided row")

    # --- resolve_hidden_fields: default config should not error and should
    # hide at least one known-internal column, never Date/OHLCV ---
    hidden = resolve_hidden_fields(dict(DEFAULTS), list(out.columns))
    check("Date" not in hidden, "Date was hidden (protected column)")
    check("EMA_9" in hidden, "expected default hidden_fields entry not applied")

    # --- index volume-blanking: Yahoo reports Volume=0 for Indian indices, so
    # every volume-derived field must be blanked rather than showing a
    # 0/0-derived value that looks computed but means nothing. (Regression
    # check: pandas >= 3's native string dtype broke the old `dtype == object`
    # test here, silently disabling this entire block.) ---
    df_idx = make_synthetic_ohlcv(seed=7)
    df_idx["Volume"] = 0.0
    out_idx = add_indicators(df_idx, cfg=dict(cfg), benchmarks=None, is_index=True)
    check((out_idx["Vol_state"] == "").all(),
          "Vol_state not blanked for a zero-volume index (dtype-check regression)")
    check(out_idx["VWAP"].isna().all(),
          "VWAP not blanked for a zero-volume index (dtype-check regression)")
    check((out_idx["Final_decision"] == "").all(),
          "Final_decision should be blank for an index row")
    check(out_idx["Final_score"].isna().all(),
          "Final_score should be NaN for an index row")

    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)

    print(f"OK: add_indicators() produced {len(out.columns)} columns over "
          f"{n} synthetic rows, all invariants held.")


if __name__ == "__main__":
    main()
