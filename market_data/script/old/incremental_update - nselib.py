import pandas as pd
from pathlib import Path
from nselib import capital_market
import datetime, time

BASE = Path(r"C:\Data\Trading\StockData\market_data")   # same absolute path
RAW = BASE / "raw" / "equity"

symbols = pd.read_csv(BASE / "config" / "symbols.csv")
symbols = symbols[symbols.active == 1]

to_date = datetime.date.today()
log = []

NUM_COLS = ["OpenPrice", "HighPrice", "LowPrice", "ClosePrice",
            "LastPrice", "PrevClose", "TotalTradedQuantity",
            "TurnoverInRs", "DeliverableQty"]

def clean(df):
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y")
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = (df[c].astype(str)
                          .str.replace(",", "", regex=False)
                          .replace({"-": None, "": None})
                          .astype(float))
    return df

for _, row in symbols.iterrows():
    sym = row["symbol"]
    fpath = RAW / f"{sym}.csv"
    try:
        if fpath.exists():
            old = pd.read_csv(fpath, parse_dates=["Date"])
            last = old["Date"].max().date()
            start = last + datetime.timedelta(days=1)
        else:
            old = pd.DataFrame()
            start = to_date - datetime.timedelta(days=3*365)

        # Guard: skip if there's no completed day to fetch
        if start >= to_date:
            print(f"– {sym}: up to date (nothing new)")
            continue

        new = capital_market.price_volume_and_deliverable_position_data(
            symbol=sym,
            from_date=start.strftime("%d-%m-%Y"),
            to_date=to_date.strftime("%d-%m-%Y"))

        if len(new):
            new = clean(new)
            combined = (pd.concat([old, new])
                          .drop_duplicates(subset="Date")
                          .sort_values("Date"))
            combined.to_csv(fpath, index=False)
            log.append({"symbol": sym,
                        "last_date": combined["Date"].max(),
                        "rows": len(combined),
                        "updated_at": datetime.datetime.now()})
            print(f"✓ {sym}: +{len(new)} rows")
        time.sleep(1)
    except Exception as e:
        print(f"✗ {sym}: {e}")

if log:
    pd.DataFrame(log).to_csv(BASE / "metadata" / "update_log.csv",
                             mode="a", header=False, index=False)