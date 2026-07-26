#!/usr/bin/env python3
"""Build a single ranked summary of every symbol's latest TA snapshot.
======================================================================
Reads BASE/config/symbols.csv for the symbol list (same active/kind
conventions as compute_ta.py), pulls data from each symbol's
BASE/ta/equity/SYMBOL_TA.csv, and writes one compact ranked file to
BASE/ta/symbol_ranking.csv.

Deliberately minimal -- this is a scan/screen view, not the TA file itself.
Open the SYMBOL_TA.csv for the full history and every field behind these.

Columns: Rank, Symbol, Portfolio (from symbols.csv), Trend_score, Final_score,
Final_score_category, Old_Final_score, Old_Final_score_category, RS_agreement,
RSI_14, RSI_state, EMA_score, Below_EMA_200, ST_state (Supertrend status),
Momentum_band, ROC_composite, Volatility_band, Vol_trend, Stage
(Stage_name/substage/confidence/days collapsed into one field),
Final_description.

ROC_composite blends the 5/10/15-trading-day % change (ROC_5/ROC_10 come
straight from the TA file; ROC_15 is derived here from Close, since
compute_ta.py's default roc_periods doesn't include 15). Old_Final_score
blends Final_score AS OF 5/10/15 sessions ago, i.e. a recency-weighted look at
where the call stood recently. Both blends weight the shorter lookback
highest (5 > 10 > 15) via RECENCY_WEIGHTS below, and both renormalize over
whichever lookbacks actually have enough history (same "missing component
shifts weight to the rest" convention as compute_ta.py's composite scores).

Ranked by Final_score, descending.

USAGE
-----
    python rank_symbols.py
    python rank_symbols.py --base "C:/Data/Trading/StockData/market_data"
    python rank_symbols.py --all   # ignore the active flag, rank every symbol
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from compute_ta import DEFAULT_BASE, normalize_symbols_frame

# Pulled directly from each SYMBOL_TA.csv's last row. Stage_* is merged into
# one "Stage" column (see load_latest_row); ROC_composite and Old_Final_score
# are built separately below since they need more than the last row alone.
_SOURCE_COLS = [
    "Trend_score", "Final_score", "RS_agreement",
    "RSI_14", "RSI_state", "EMA_score", "Below_EMA_200", "ST_state",
    "Momentum_band", "Volatility_band", "Vol_trend",
    "Stage_name", "Stage_substage", "Stage_confidence", "Stage_days",
    "Final_description",
]

# Sorted on this column, descending -- the field a bar can score highest on
# only once every decision gate (stage/trend/momentum/volume/volatility) has
# actually passed, so it ranks "cleanest setups first" rather than just
# "strongest trend first".
SORT_COLUMN = "Final_score"

# Lookbacks (trading days) for both composites below, and how much each one
# counts -- the shorter the lookback, the more weight, so a composite reacts
# mainly to what just happened rather than being dragged down by 3-week-old
# history. Must be sorted longest-last; renormalization (see _weighted_avg)
# handles a symbol too short for the 15-day leg.
RECENCY_WEIGHTS = {5: 0.5, 10: 0.3, 15: 0.2}

# Letter-grade cut points for Final_score / Old_Final_score, highest first.
# A is the top band; anything below the lowest threshold is E. These are a
# plain 0-100 split (80/65/50/35), not tied to the decision-gate ceilings
# inside compute_ta.py -- adjust freely if a different grading feels right.
FINAL_SCORE_BANDS = [
    (80.0, "A"),
    (65.0, "B"),
    (50.0, "C"),
    (35.0, "D"),
    (float("-inf"), "E"),
]

OUTPUT_COLUMNS = [
    "Rank", "Symbol", "Portfolio", "Trend_score",
    "Final_score", "Final_score_category",
    "Old_Final_score", "Old_Final_score_category",
    "RS_agreement", "RSI_14", "RSI_state", "EMA_score", "Below_EMA_200",
    "ST_state", "Momentum_band", "ROC_composite",
    "Volatility_band", "Vol_trend", "Stage", "Final_description",
]


def _format_stage(name, substage, confidence, days):
    """Collapse Stage_name/substage/confidence/days into one readable field,
    e.g. "2 - Advancing (mid, conf 72, 15d)". Blank wherever no stage call
    exists yet (warm-up, or too little history for stage classification)."""
    if pd.isna(name) or str(name).strip() == "":
        return ""
    parts = []
    if not pd.isna(substage) and str(substage).strip():
        parts.append(str(substage))
    if not pd.isna(confidence):
        parts.append(f"conf {float(confidence):.0f}")
    if not pd.isna(days):
        parts.append(f"{int(days)}d")
    return f"{name} ({', '.join(parts)})" if parts else str(name)


def _weighted_avg(values_by_lookback, weights):
    """Weighted average of {lookback: value}, renormalized over whichever
    lookbacks have a real (non-NaN) value -- a symbol with less than 15
    trading days of history simply drops that leg rather than blanking the
    whole composite. NaN if nothing is available at all."""
    total = 0.0
    wsum = 0.0
    for lookback, value in values_by_lookback.items():
        if value is None or pd.isna(value):
            continue
        w = weights.get(lookback, 0.0)
        total += float(value) * w
        wsum += w
    return round(total / wsum, 2) if wsum > 0 else np.nan


def _score_category(score):
    """A/B/C/D/E letter grade for a 0-100 score, A = highest. Blank if the
    score itself isn't available (e.g. too little history)."""
    if pd.isna(score):
        return ""
    for threshold, label in FINAL_SCORE_BANDS:
        if score >= threshold:
            return label
    return ""


def _value_n_sessions_ago(series, n):
    """series.iloc[-1] is "today"; return the value from `n` trading
    sessions before that, or NaN if the file doesn't go back that far."""
    idx = len(series) - 1 - n
    return series.iloc[idx] if idx >= 0 else np.nan


def load_latest_row(ta_path: Path, symbol: str, portfolio: str):
    """Return a dict of the summary fields for one symbol, or None if the TA
    file is missing/unusable.

    Missing source columns (e.g. a customized hidden_fields dropped one) fall
    back to NaN/blank rather than aborting the whole run over one symbol.
    `portfolio` comes from symbols.csv, not the TA file, and is passed
    straight through."""
    if not ta_path.exists():
        print(f"  skip {symbol}: TA file not found ({ta_path.name})")
        return None
    try:
        df = pd.read_csv(ta_path)
    except Exception as e:
        print(f"  skip {symbol}: failed to read TA file ({type(e).__name__}: {e})")
        return None
    if df.empty:
        print(f"  skip {symbol}: TA file is empty")
        return None

    last = df.iloc[-1]
    row = {"Symbol": symbol, "Portfolio": portfolio}
    for col in _SOURCE_COLS:
        row[col] = last[col] if col in df.columns else np.nan

    row["Stage"] = _format_stage(row.pop("Stage_name"), row.pop("Stage_substage"),
                                 row.pop("Stage_confidence"), row.pop("Stage_days"))

    # ---- ROC_composite: 5/10/15-day % change, 5-day weighted highest -------
    close = pd.to_numeric(df["Close"], errors="coerce") if "Close" in df.columns else pd.Series(dtype=float)
    roc_by_lookback = {}
    for lookback in RECENCY_WEIGHTS:
        col = f"ROC_{lookback}"
        if col in df.columns:
            roc_by_lookback[lookback] = last[col]
        else:
            # Not precomputed by compute_ta.py (e.g. ROC_15 isn't in the
            # default roc_periods) -> derive it directly from Close.
            past = _value_n_sessions_ago(close, lookback)
            roc_by_lookback[lookback] = (
                (close.iloc[-1] / past - 1.0) * 100.0
                if not pd.isna(past) and past != 0 else np.nan)
    row["ROC_composite"] = _weighted_avg(roc_by_lookback, RECENCY_WEIGHTS)

    # ---- Old_Final_score: Final_score AS OF 5/10/15 sessions ago -----------
    if "Final_score" in df.columns:
        fscore_series = pd.to_numeric(df["Final_score"], errors="coerce")
        fscore_by_lookback = {lb: _value_n_sessions_ago(fscore_series, lb)
                              for lb in RECENCY_WEIGHTS}
    else:
        fscore_by_lookback = {lb: np.nan for lb in RECENCY_WEIGHTS}
    row["Old_Final_score"] = _weighted_avg(fscore_by_lookback, RECENCY_WEIGHTS)

    row["Final_score_category"] = _score_category(row.get("Final_score"))
    row["Old_Final_score_category"] = _score_category(row["Old_Final_score"])

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE),
                    help="market_data base folder (contains config/, ta/)")
    ap.add_argument("--all", action="store_true",
                    help="rank every symbol in symbols.csv, ignoring the active flag")
    ap.add_argument("--out", default="symbol_ranking.csv",
                    help="output filename, written under BASE/ta (default: symbol_ranking.csv)")
    args = ap.parse_args()

    base = Path(args.base)
    sym_csv = base / "config" / "symbols.csv"
    ta_dir = base / "ta" / "equity"
    out_path = base / "ta" / args.out

    if not sym_csv.exists():
        sys.exit(f"Symbol list not found: {sym_csv}")
    if not ta_dir.exists():
        sys.exit(f"TA folder not found: {ta_dir}")

    symbols = pd.read_csv(sym_csv)
    if "symbol" not in symbols.columns:
        sys.exit(f"'symbol' column missing in {sym_csv}")
    symbols = normalize_symbols_frame(symbols)

    if not args.all and "active" in symbols.columns:
        symbols = symbols[symbols.active == 1]

    # Benchmark/index rows carry no Final_description or RS_agreement (RS
    # against yourself is meaningless), so ranking them alongside tradable
    # equities would just be noise -- they're excluded here.
    symbols = symbols[symbols["kind"] == "equity"]

    sym_list = symbols["symbol"].astype(str).tolist()
    if not sym_list:
        sys.exit("No symbols to rank (check the active flag or use --all).")

    if "portfolio" in symbols.columns:
        portfolios = dict(zip(symbols["symbol"].astype(str), symbols["portfolio"].astype(str)))
    else:
        portfolios = {}

    print(f"Ranking {len(sym_list)} symbols\n  ta in : {ta_dir}\n  out   : {out_path}\n")

    rows = []
    for sym in sym_list:
        row = load_latest_row(ta_dir / f"{sym}_TA.csv", sym, portfolios.get(sym, ""))
        if row is not None:
            rows.append(row)

    if not rows:
        sys.exit("No TA data could be read -- run compute_ta.py first.")

    result = pd.DataFrame(rows)
    result[SORT_COLUMN] = pd.to_numeric(result[SORT_COLUMN], errors="coerce")
    result = result.sort_values(SORT_COLUMN, ascending=False, na_position="last")
    result.insert(0, "Rank", range(1, len(result) + 1))
    result = result[OUTPUT_COLUMNS]
    result.to_csv(out_path, index=False)

    print(f"Done: {len(result)} symbols ranked -> {out_path}")


if __name__ == "__main__":
    main()
