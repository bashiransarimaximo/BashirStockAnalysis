"""Incrementally top up OHLCV for every active symbol in config/symbols.csv.

Shares its symbols.csv conventions with bulk_load.py and compute_ta.py:
  * whitespace stripped from every column and value
  * 'active' coerced to an integer
  * optional 'kind' column (equity | index); index rows use Yahoo '^' codes
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

BASE = Path(r"C:\Data\Trading\StockData\market_data")
RAW = BASE / "raw" / "equity"           # same folder as bulk_load.py
RAW.mkdir(parents=True, exist_ok=True)

FULL_YEARS = 5          # matches bulk_load.py, so a re-download is not shorter
ADJ_TOLERANCE = 0.001   # >0.1% change in a stored Close => corporate action

# Indices are NOT ".NS" tickers -- they use Yahoo's "^" index codes.
# Keep this map identical to bulk_load.py.
INDEX_TICKERS = {
    "NIFTY50":        "^NSEI",       # Nifty 50
    "NIFTY500":       "^CRSLDX",     # Nifty 500 (broad market)
    "NIFTYBANK":      "^NSEBANK",    # Bank Nifty
    "NIFTYSMLCAP250": "^CNXSC",      # Nifty Smallcap
    "NIFTYMIDCAP150": "^NSEMDCP50",  # verify before relying on this one
    "SENSEX":         "^BSESN",
    "INDIAVIX":       "^INDIAVIX",
}

# Log schema is FIXED and shared with bulk_load.py. This script appends with
# header=False, so any column-order drift between the two would silently write
# values into the wrong columns of an existing log.
LOG_COLUMNS = ["symbol", "kind", "yf_ticker", "last_date", "rows",
               "action", "updated_at"]


def normalize_symbols_frame(symbols):
    """Clean symbols.csv exactly as bulk_load.py and compute_ta.py do.

    A trailing space ("NIFTY50 ,1,index ") is invisible in an editor but yields
    "NIFTY50 .csv" on disk, which compute_ta.py never finds. `active` is coerced
    because a padded "1 " makes `active == 1` false for every row -- which
    silently updates nothing at all."""
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


def fetch(ticker, start, end):
    """Download adjusted OHLCV from Yahoo, keeping native column names."""
    df = yf.download(ticker, start=start,
                     end=end + datetime.timedelta(days=1),   # end is exclusive
                     auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]   # flatten single-ticker columns
    df["Date"] = pd.to_datetime(df["Date"])
    return df.drop_duplicates(subset="Date").sort_values("Date")


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
    full_start = to_date - datetime.timedelta(days=FULL_YEARS * 365)
    log = []

    for _, row in symbols.iterrows():
        sym = row["symbol"]
        kind = row["kind"]
        yf_ticker = resolve_ticker(sym, kind, row["index"])
        if yf_ticker is None:
            continue

        fpath = RAW / f"{sym}.csv"      # stripped stem, matches compute_ta.py
        try:
            if not fpath.exists():
                # first time for this symbol -> full history
                combined = fetch(yf_ticker, full_start, to_date)
                action = "created"
            else:
                old = pd.read_csv(fpath, parse_dates=["Date"])
                old.columns = [str(c).strip() for c in old.columns]
                if "Date" not in old.columns or "Close" not in old.columns:
                    print(f"X  {sym}: existing file lacks Date/Close -> skipped")
                    continue

                last = old["Date"].max().date()
                start = last + datetime.timedelta(days=1)

                # Guard: skip if there's no completed day to fetch
                if start >= to_date:
                    print(f"-  {sym}: up to date (nothing new)")
                    continue

                new = fetch(yf_ticker, start, to_date)
                if new.empty:
                    print(f"-  {sym}: no new rows")
                    continue

                # --- adjustment check ---
                # If Yahoo's Close for a date we already saved has changed, a
                # corporate action re-adjusted history -> re-download in full.
                # Indices are never split/bonus adjusted, so skip the extra call.
                adjusted_happened = False
                if kind != "index":
                    recheck = fetch(yf_ticker,
                                    last - datetime.timedelta(days=5), last)
                    if not recheck.empty and last in set(old["Date"].dt.date):
                        old_close = float(old.loc[old["Date"].dt.date == last,
                                                  "Close"].iloc[0])
                        yh = recheck.loc[recheck["Date"].dt.date == last, "Close"]
                        if len(yh) and old_close:
                            new_close = float(yh.iloc[0])
                            if abs(new_close - old_close) / old_close > ADJ_TOLERANCE:
                                adjusted_happened = True

                if adjusted_happened:
                    print(f"~  {sym}: corporate action detected -> full re-download")
                    combined = fetch(yf_ticker, full_start, to_date)
                    action = "readjusted"
                else:
                    combined = (pd.concat([old, new])
                                  .drop_duplicates(subset="Date")
                                  .sort_values("Date"))
                    action = f"+{len(new)} rows"

            if combined.empty:
                print(f"X  {sym}: no data returned (check ticker {yf_ticker})")
                continue

            # Indices carry no traded volume on Yahoo. compute_ta.py detects
            # this and blanks the volume-derived fields rather than emitting
            # ratios of zeros.
            note = ""
            if kind == "index" and "Volume" in combined.columns:
                if (combined["Volume"].fillna(0) == 0).mean() > 0.5:
                    note = "   [Volume is 0 -- normal for an index]"

            combined.to_csv(fpath, index=False)
            log.append({"symbol": sym, "kind": kind, "yf_ticker": yf_ticker,
                        "last_date": combined["Date"].max(),
                        "rows": len(combined), "action": action,
                        "updated_at": datetime.datetime.now()})
            print(f"OK {sym} ({yf_ticker}): {action}{note}")
            time.sleep(1)
        except Exception as e:
            print(f"X  {sym} ({yf_ticker}): {type(e).__name__}: {e}")

    if log:
        (BASE / "metadata").mkdir(exist_ok=True)
        log_path = BASE / "metadata" / "update_log.csv"
        df_log = pd.DataFrame(log).reindex(columns=LOG_COLUMNS)

        # Append only when the existing file already uses this schema. A log
        # written by the OLD script has different columns, and appending under
        # header=False would shift every value into the wrong column.
        write_header, mode = True, "w"
        if log_path.exists():
            try:
                existing = pd.read_csv(log_path, nrows=0)
                if [c.strip() for c in existing.columns] == LOG_COLUMNS:
                    write_header, mode = False, "a"
                else:
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup = log_path.with_name(f"update_log_{stamp}.csv")
                    log_path.rename(backup)
                    print(f"!  update_log.csv used an old schema -> archived as "
                          f"{backup.name}, starting a new log")
            except Exception:
                pass
        df_log.to_csv(log_path, mode=mode, header=write_header, index=False)

    # Warn about any benchmark with no data file at all: RS_* degrades silently
    # in compute_ta.py otherwise, and a blank RS column looks the same as a
    # stock that simply has no relative strength.
    updated = {r["symbol"] for r in log}
    for _, r in symbols[symbols.kind == "index"].iterrows():
        if r["symbol"] not in updated and not (RAW / f"{r['symbol']}.csv").exists():
            print(f"!  benchmark {r['symbol']} has no data file -> "
                  f"RS_{r['symbol']}_* will be blank in compute_ta.py")

    print(f"\nDone: {len(log)} symbol(s) updated in {RAW}")


if __name__ == "__main__":
    main()
