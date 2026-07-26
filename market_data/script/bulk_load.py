"""Download OHLCV for every active symbol in config/symbols.csv.

Shares its symbols.csv conventions with compute_ta.py:
  * whitespace is stripped from every column and value
  * 'active' is coerced to an integer
  * optional 'kind' column (equity | index); index rows use Yahoo '^' codes
    instead of an equity suffix
  * optional 'index' column (NS | BO) picks the Yahoo exchange suffix for
    equity rows (.NS / .BO); ignored for kind=index rows. Blank or
    unrecognised values fall back to NS.

BASE/config/symbols.csv     -> symbol, active, [kind], [index]
BASE/raw/equity/SYMBOL.csv  -> Date, Open, High, Low, Close, Volume
BASE/metadata/update_log.csv
"""

import datetime
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# Resolved relative to this file (script/../ = market_data/) rather than
# hardcoded, so the same code works unmodified in any clone location --
# including a CI runner, where C:\Data\... simply doesn't exist.
BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "raw" / "equity"
RAW.mkdir(parents=True, exist_ok=True)

YEARS_BACK = 5

# Indices are NOT ".NS" tickers -- they use Yahoo's "^" index codes. Anything
# marked kind=index must appear here or it is skipped with a clear message.
INDEX_TICKERS = {
    "NIFTY50":        "^NSEI",       # Nifty 50
    "NIFTY500":       "^CRSLDX",     # Nifty 500 (broad market: large+mid+small)
    "NIFTYBANK":      "^NSEBANK",    # Bank Nifty
    "NIFTYSMLCAP250": "^CNXSC",      # Nifty Smallcap
    "NIFTYMIDCAP150": "^NSEMDCP50",  # verify before relying on this one
    "SENSEX":         "^BSESN",
    "INDIAVIX":       "^INDIAVIX",
}


def normalize_symbols_frame(symbols):
    """Clean symbols.csv exactly as compute_ta.py does.

    A trailing space ("NIFTY50 ,1,index ") is invisible in a text editor but
    produces "NIFTY50 .csv" on disk, which compute_ta.py will never find. Both
    scripts must strip identically or they disagree about what a symbol is
    called. `active` is coerced because a padded "1 " makes `active == 1` false
    for every row -- which silently downloads nothing at all."""
    symbols = symbols.copy()
    symbols.columns = [str(c).strip() for c in symbols.columns]
    for col in symbols.columns:
        if not pd.api.types.is_numeric_dtype(symbols[col]):
            symbols[col] = symbols[col].astype(str).str.strip()

    if "active" in symbols.columns:
        symbols["active"] = (pd.to_numeric(symbols["active"], errors="coerce")
                             .fillna(0).astype(int))

    if "kind" not in symbols.columns:
        symbols["kind"] = "equity"
    symbols["kind"] = (symbols["kind"].astype(str).str.strip().str.lower()
                       .replace({"": "equity", "nan": "equity", "none": "equity"}))

    known = {"equity", "index"}
    bad = sorted(set(symbols["kind"]) - known)
    if bad:
        print(f"  symbols: unrecognised kind value(s) {bad} -> treated as equity")
        symbols.loc[~symbols["kind"].isin(known), "kind"] = "equity"

    # 'index' column: which exchange an EQUITY row's ticker lives on (NS/BO).
    # The generic strip loop above already ran astype(str) on it, which turns
    # a blank cell into the literal string "nan" -- undo that here, the same
    # way 'kind' does, so resolve_ticker sees a real blank rather than "NAN".
    if "index" not in symbols.columns:
        symbols["index"] = ""
    symbols["index"] = (symbols["index"].astype(str).str.strip().str.upper()
                        .replace({"NAN": "", "NONE": ""}))
    return symbols


DEFAULT_EXCHANGE = "NS"
KNOWN_EXCHANGES = {"NS", "BO"}


def resolve_ticker(sym, kind, exchange):
    """Map a symbol stem to its Yahoo ticker.

    `exchange` ('NS' or 'BO', from symbols.csv's 'index' column) picks the
    Yahoo suffix for an EQUITY row. Blank or unrecognised values fall back to
    NS with a warning rather than silently guessing. Index rows ignore
    `exchange` entirely -- they use the INDEX_TICKERS '^' codes instead."""
    if kind == "index":
        t = INDEX_TICKERS.get(sym)
        if t is None:
            print(f"!  {sym}: kind=index but no Yahoo code mapped -> skipped. "
                  f"Add it to INDEX_TICKERS.")
        return t
    if exchange not in KNOWN_EXCHANGES:
        if exchange:
            print(f"!  {sym}: unrecognised exchange '{exchange}' in the 'index' "
                  f"column -> defaulting to {DEFAULT_EXCHANGE}")
        else:
            print(f"!  {sym}: no exchange in the 'index' column -> "
                  f"defaulting to {DEFAULT_EXCHANGE}")
        exchange = DEFAULT_EXCHANGE
    return f"{sym}.{exchange}"


def main():
    symbols = pd.read_csv(BASE / "config" / "symbols.csv")
    if "symbol" not in symbols.columns:
        raise SystemExit("'symbol' column missing in symbols.csv")

    symbols = normalize_symbols_frame(symbols)
    if "active" in symbols.columns:
        symbols = symbols[symbols.active == 1]
    if symbols.empty:
        raise SystemExit("No active symbols (check the active flag).")

    to_date = datetime.date.today()
    from_date = to_date - datetime.timedelta(days=YEARS_BACK * 365)

    log = []
    for _, row in symbols.iterrows():
        sym = row["symbol"]
        kind = row["kind"]
        yf_ticker = resolve_ticker(sym, kind, row["index"])
        if yf_ticker is None:
            continue

        try:
            # auto_adjust=True -> OHLC adjusted for splits, bonuses & dividends.
            # Without it, a split puts a phantom vertical drop in the series and
            # every long-lookback indicator breaks at that date.
            df = yf.download(yf_ticker,
                             start=from_date,
                             end=to_date + datetime.timedelta(days=1),  # exclusive
                             auto_adjust=True,
                             progress=False)

            if df.empty:
                print(f"X  {sym}: no data returned (check ticker {yf_ticker})")
                continue

            # yfinance returns a DatetimeIndex + sometimes multi-level columns
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            df["Date"] = pd.to_datetime(df["Date"])
            df = df.drop_duplicates(subset="Date").sort_values("Date")

            # Indices carry no traded volume on Yahoo. compute_ta.py detects this
            # and blanks the volume-derived fields rather than emitting zeros.
            note = ""
            if kind == "index" and "Volume" in df.columns:
                if (df["Volume"].fillna(0) == 0).mean() > 0.5:
                    note = "   [Volume is 0 -- normal for an index]"

            # Filename uses the STRIPPED symbol, matching compute_ta.py's lookup.
            df.to_csv(RAW / f"{sym}.csv", index=False)
            log.append({"symbol": sym, "kind": kind, "yf_ticker": yf_ticker,
                        "last_date": df["Date"].max(), "rows": len(df),
                        "updated_at": datetime.datetime.now()})
            print(f"OK {sym} ({yf_ticker}): {len(df)} rows{note}")
            time.sleep(1)
        except Exception as e:
            print(f"X  {sym} ({yf_ticker}): {type(e).__name__}: {e}")

    (BASE / "metadata").mkdir(exist_ok=True)
    pd.DataFrame(log).to_csv(BASE / "metadata" / "update_log.csv", index=False)

    # Fail loudly on a missing benchmark: RS_* degrades silently otherwise, and a
    # blank RS column looks the same as a stock with no relative strength.
    got = {r["symbol"] for r in log}
    missing = [r["symbol"] for _, r in symbols[symbols.kind == "index"].iterrows()
               if r["symbol"] not in got]
    for m in missing:
        print(f"!  benchmark {m} NOT downloaded -> RS_{m}_* will be blank in compute_ta.py")

    print(f"\nDone: {len(log)}/{len(symbols)} symbols written to {RAW}")


if __name__ == "__main__":
    main()
