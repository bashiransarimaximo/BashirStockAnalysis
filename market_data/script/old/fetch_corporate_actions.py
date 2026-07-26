import pandas as pd
from pathlib import Path
from nselib import capital_market
import re, datetime, time

BASE = Path(r"C:\Data\Trading\StockData\market_data")
CONFIG = BASE / "config"
CONFIG.mkdir(parents=True, exist_ok=True)
OUT = CONFIG / "corporate_actions.csv"

symbols = pd.read_csv(CONFIG / "symbols.csv")
symbols = symbols[symbols.active == 1]["symbol"].tolist()


# ---------------------------------------------------------------------------
#  Factor parsing
#  factor = old_shares / new_shares  (always <= 1)
#  Pre-ex-date prices are MULTIPLIED by factor; volumes DIVIDED by factor.
# ---------------------------------------------------------------------------
def parse_factor(text):
    """Return old_shares/new_shares factor, or None if not a split/bonus."""
    t = " ".join(str(text).lower().split())   # normalize whitespace

    # ---- BONUS ----  "a:b" = a new shares for every b held
    # Tolerates: "bonus 1:1", "bonus issue 1:1", "bonus (1:1)", "bonus- 2:1"
    m = re.search(r"bonus\b[^0-9]*(\d+)\s*:\s*(\d+)", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a + b > 0:
            return b / (a + b)      # 1:1→0.5 ; 2:1→0.333 ; 3:2→0.4

    # ---- SPLIT / SUB-DIVISION ----
    # Primary: face-value phrasing, anchored on Rs/Re/FV markers so stray
    # counts ("1 share ... into 5 shares") don't get mistaken for face values.
    # e.g. "face value split from rs 10 to rs 2", "sub-division re 10 to re 1"
    m = re.search(
        r"(?:split|sub[\s-]*divi\w*).*?"
        r"(?:rs\.?|re\.?|₹|f\.?v\.?|face\s*value)\s*(\d+)"
        r".*?(?:rs\.?|re\.?|₹)\s*(\d+)", t)
    if m:
        old_fv, new_fv = int(m.group(1)), int(m.group(2))
        if new_fv and new_fv < old_fv:
            return new_fv / old_fv  # 10→5=0.5 ; 10→2=0.2 ; 10→1=0.1

    # Fallback: explicit split ratio, "split 5:1" / "stock split 2 : 1"
    m = re.search(r"(?:split|sub[\s-]*divi\w*)[^0-9]*(\d+)\s*:\s*(\d+)", t)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        if x and y:
            return min(x, y) / max(x, y)   # normalize to <=1 factor

    return None                     # dividends, AGMs, rights, etc. → ignored


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def pick_col(cols, must_have, label):
    """Find one column containing all substrings in must_have; raise if missing."""
    hits = [c for c in cols if all(k in c.lower() for k in must_have)]
    if not hits:
        raise KeyError(f"no column for {label} (need {must_have}) in {list(cols)}")
    return hits[0]


def fetch_all_actions(start_year=2020):
    """Fetch corporate actions year-by-year so nselib can't truncate a long range."""
    frames = []
    this_year = datetime.date.today().year
    for yr in range(start_year, this_year + 1):
        frm = f"01-01-{yr}"
        to  = (f"31-12-{yr}" if yr < this_year
               else datetime.date.today().strftime("%d-%m-%Y"))
        try:
            chunk = capital_market.corporate_action(from_date=frm, to_date=to)
            if len(chunk):
                frames.append(chunk)
            print(f"  · {yr}: {len(chunk)} rows")
            time.sleep(1)           # be polite to NSE
        except Exception as e:
            print(f"  ⚠ fetch {yr}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
print("Fetching corporate actions (year by year)…")
ca_all = fetch_all_actions()

if ca_all.empty:
    raise SystemExit("No corporate-action data returned — check nselib / network.")

# resolve column names once, loudly
cols = list(ca_all.columns)
symcol = (pick_col(cols, ["symbol"], "symbol")
          if any("symbol" in c.lower() for c in cols)
          else pick_col(cols, ["series"], "symbol"))
purpcol = next((c for c in cols
                if "purpose" in c.lower() or "subject" in c.lower()), None)
if purpcol is None:
    raise KeyError(f"no purpose/subject column in {cols}")
try:
    excol = pick_col(cols, ["ex", "date"], "ex-date")
except KeyError:
    excol = pick_col(cols, ["rec", "date"], "record-date")
    print(f"  ⚠ no ex-date column; using record date '{excol}' instead")

print(f"Columns → symbol='{symcol}', purpose='{purpcol}', ex_date='{excol}'")

# normalize symbol column: strip spaces, drop -EQ/-BE/-BZ suffixes
sym_series = (ca_all[symcol].astype(str).str.strip().str.upper()
                            .str.replace(r"[-\s]*(EQ|BE|BZ)$", "", regex=True))

rows, unparsed = [], []

for sym in symbols:
    sub = ca_all[sym_series == sym.upper()]
    for _, r in sub.iterrows():
        purpose = str(r[purpcol])
        f = parse_factor(purpose)
        ex = pd.to_datetime(r[excol], errors="coerce", dayfirst=True)
        if f and pd.notna(ex):
            kind = ("bonus" if "bonus" in purpose.lower()
                    else "split" if re.search(r"split|sub[\s-]*divi", purpose.lower())
                    else "other")
            rows.append({"symbol": sym, "ex_date": ex.date(),
                         "factor": round(f, 6), "type": kind,
                         "purpose": purpose})
        elif re.search(r"bonus|split|sub[\s-]*divi", purpose.lower()):
            unparsed.append({"symbol": sym, "ex_date": ex, "purpose": purpose})

df = (pd.DataFrame(rows)
        .drop_duplicates(subset=["symbol", "ex_date"])
        .sort_values(["symbol", "ex_date"]))
df.to_csv(OUT, index=False)
print(f"\n✓ wrote {len(df)} actions to {OUT}")

# ---- validation output: eyeball factors against known events ----
if len(df):
    print("\nParsed actions:")
    print(df.to_string(index=False))
    print("\nSplits only (verify face-value factors):")
    splits = df[df["type"] == "split"]
    print(splits.to_string(index=False) if len(splits) else "  (none)")

if unparsed:
    print("\n⚠ REVIEW — looked like split/bonus but factor couldn't be parsed:")
    for u in unparsed:
        print(f"   {u['symbol']}  {u['ex_date']}  {u['purpose']}")
    print("  → add these manually to corporate_actions.csv if they are real events.")