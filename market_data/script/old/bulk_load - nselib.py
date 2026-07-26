import pandas as pd
from pathlib import Path
from nselib import capital_market
import datetime, time

BASE = Path(r"C:\Data\Trading\StockData\market_data")
RAW = BASE / "raw" / "equity"
RAW.mkdir(parents=True, exist_ok=True)

symbols = pd.read_csv(BASE / "config" / "symbols.csv")
symbols = symbols[symbols.active == 1]

to_date = datetime.date.today()
from_date = to_date - datetime.timedelta(days=3*365)
from_str = from_date.strftime("%d-%m-%Y")
to_str = to_date.strftime("%d-%m-%Y")

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

log = []

for _, row in symbols.iterrows():
    sym = row["symbol"]
    try:
        df = capital_market.price_volume_and_deliverable_position_data(
            symbol=sym, from_date=from_str, to_date=to_str)
        df = clean(df)
        df = df.drop_duplicates(subset="Date").sort_values("Date")

        df.to_csv(RAW / f"{sym}.csv", index=False)
        log.append({"symbol": sym, "last_date": df["Date"].max(),
                    "rows": len(df),
                    "updated_at": datetime.datetime.now()})
        print(f"✓ {sym}: {len(df)} rows")
        time.sleep(1)
    except Exception as e:
        print(f"✗ {sym}: {e}")

(BASE / "metadata").mkdir(exist_ok=True)
pd.DataFrame(log).to_csv(BASE / "metadata" / "update_log.csv", index=False)