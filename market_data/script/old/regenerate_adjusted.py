import pandas as pd
from pathlib import Path
from nselib import capital_market
import re, datetime

BASE = Path(r"C:\Data\Trading\StockData\market_data")
CONFIG = BASE / "config"
CONFIG.mkdir(parents=True, exist_ok=True)
OUT = CONFIG / "corporate_actions.csv"

symbols = pd.read_csv(CONFIG / "symbols.csv")
symbols = symbols[symbols.active == 1]["symbol"].tolist()

def parse_factor(text):
    """Return old_shares/new_shares factor, or None if not a split/bonus."""
    t = text.lower()

    # BONUS  e.g. "bonus 1:1", "bonus 2:1"  → a new shares for every b held
    m = re.search(r"bonus\s+(\d+)\s*:\s*(\d+)", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return b / (a + b)          # 1:1 → 1/2 = 0.5 ; 2:1 → 1/3 = 0.333

    # SPLIT  e.g. "split ... from rs 10 to rs 5"  → face value ratio
    m = re.search(r"(?:split|sub-?divi).*?(\d+).*?(?:to|/)\s*(?:rs\.?\s*)?(\d+)", t)
    if m:
        old_fv, new_fv = int(m.group(1)), int(m.group(2))
        if new_fv and new_fv < old_fv:
            return new_fv / old_fv  # 10→5 = 0.5 ; 10→2 = 0.2

    return None                     # dividends, AGMs, etc. → ignored

rows, unparsed = [], []
today = datetime.date.today().strftime("%d-%m-%Y")

for sym in symbols:
    try:
        ca = capital_market.corporate_action(from_date="01-01-2020",
                                              to_date=today)
        # filter to this symbol (column name may be 'symbol' or 'Symbol')
        symcol = "symbol" if "symbol" in ca.columns else "Symbol"
        purpcol = [c for c in ca.columns if "purpose" in c.lower() or "subject" in c.lower()][0]
        excol   = [c for c in ca.columns if "ex" in c.lower() and "date" in c.lower()][0]

        sub = ca[ca[symcol].str.upper() == sym.upper()]
        for _, r in sub.iterrows():
            purpose = str(r[purpcol])
            f = parse_factor(purpose)
            ex = pd.to_datetime(r[excol], errors="coerce")
            if f and pd.notna(ex):
                rows.append({"symbol": sym, "ex_date": ex.date(),
                             "factor": round(f, 6), "purpose": purpose})
            elif re.search(r"bonus|split|sub-?divi", purpose.lower()):
                unparsed.append({"symbol": sym, "ex_date": ex,
                                 "purpose": purpose})   # looks relevant but failed
    except Exception as e:
        print(f"✗ {sym}: {e}")

df = pd.DataFrame(rows).drop_duplicates(subset=["symbol", "ex_date"])
df.to_csv(OUT, index=False)
print(f"✓ wrote {len(df)} actions to {OUT}")

if unparsed:
    print("\n⚠ REVIEW — looked like split/bonus but couldn't parse the factor:")
    for u in unparsed:
        print(f"   {u['symbol']}  {u['ex_date']}  {u['purpose']}")