import time
import requests
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
# Polymarket Subgraph endpoint (official docs page lists it; if this specific URL differs,
# we will replace it with the one you see in the docs page)
SUBGRAPH_URL = "https://api.thegraph.com/subgraphs/name/polymarket/polymarket"  # common pattern

KEYWORDS = ["taiwan", "gaza", "ceasefire", "ukraine", "russia", "sanctions", "iran", "election", "war"]

MAX_MARKETS_TOTAL = 40      # research-grade = fewer markets, deeper history
MIN_TRADES = 500            # minimum raw records per market to be usable
RESAMPLE_FREQ = "1h"
K_STEPS = 3
H_STEPS = 12

REQUEST_SLEEP = 0.2


# -----------------------------
# Econometrics helpers
# -----------------------------
def demean_by_group(df, group_col, cols):
    g = df.groupby(group_col)[cols].transform("mean")
    out = df.copy()
    for c in cols:
        out[c] = out[c] - g[c]
    return out

def ols_cluster(y, X, groups):
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": groups})


# -----------------------------
# Subgraph helpers
# -----------------------------
def gql(query, variables=None):
    payload = {"query": query, "variables": variables or {}}
    r = requests.post(SUBGRAPH_URL, json=payload, timeout=60)
    r.raise_for_status()
    out = r.json()
    if "errors" in out:
        raise RuntimeError(out["errors"])
    return out["data"]

def keyword_match(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in KEYWORDS)

def list_markets(limit=200, skip=0):
    """
    NOTE: Subgraph schemas differ. This is a *template* query.
    If it errors, paste the error and I'll adapt the fields to the actual schema quickly.
    """
    q = """
    query($first:Int!, $skip:Int!) {
      markets(first:$first, skip:$skip, orderBy:createdAt, orderDirection:desc) {
        id
        question
        liquidity
        volume
      }
    }
    """
    return gql(q, {"first": limit, "skip": skip})["markets"]

def list_trades(market_id, first=1000, skip=0):
    """
    Template: trades/fills associated with a market.
    Schema differs; we’ll adapt based on the error message if needed.
    """
    q = """
    query($m:String!, $first:Int!, $skip:Int!) {
      trades(first:$first, skip:$skip, orderBy:timestamp, orderDirection:asc, where:{market:$m}) {
        id
        timestamp
        price
        amount
        trader
      }
    }
    """
    return gql(q, {"m": market_id, "first": first, "skip": skip})["trades"]


def trades_to_price_series(trades, freq="1h"):
    df = pd.DataFrame(trades)
    if df.empty:
        return None

    # timestamps in subgraphs are usually seconds (string)
    df["t"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="s", utc=True)
    df["p"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["t", "p"]).sort_values("t")

    if df.empty:
        return None

    # last traded price per hour, forward-filled
    s = df.set_index("t")["p"].resample(freq).last().ffill()
    out = s.reset_index().rename(columns={"p": "p"})
    out["t"] = pd.to_datetime(out["t"], utc=True)
    return out


def build_panel(prob_df, k_steps, h_steps):
    p = prob_df["p"].to_numpy()
    t = prob_df["t"].reset_index(drop=True)

    if len(p) <= (k_steps + h_steps + 5):
        return None

    idx = np.arange(k_steps, len(p) - h_steps)
    dp0 = p[idx] - p[idx - k_steps]
    dph = p[idx + h_steps] - p[idx]

    return pd.DataFrame({"t": t.iloc[idx].to_numpy(), "dp0": dp0, "dph": dph})


# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Discover markets
    candidates = []
    skip = 0
    while len(candidates) < MAX_MARKETS_TOTAL and skip < 2000:
        batch = list_markets(limit=200, skip=skip)
        if not batch:
            break
        for m in batch:
            if keyword_match(m.get("question", "")):
                candidates.append(m)
                if len(candidates) >= MAX_MARKETS_TOTAL:
                    break
        skip += 200
        time.sleep(REQUEST_SLEEP)

    print("Candidate markets:", len(candidates))
    if not candidates:
        print("No markets matched keywords. Add broader KEYWORDS.")
        return

    # 2) Pull trades and build panels
    panels = []
    kept = 0

    for m in tqdm(candidates):
        mid = m["id"]
        all_trades = []
        sk = 0
        while sk < 20000:  # cap pagination
            chunk = list_trades(mid, first=1000, skip=sk)
            if not chunk:
                break
            all_trades.extend(chunk)
            sk += 1000
            time.sleep(REQUEST_SLEEP)

        if len(all_trades) < MIN_TRADES:
            continue

        ps = trades_to_price_series(all_trades, freq=RESAMPLE_FREQ)
        if ps is None or len(ps) < 200:
            continue

        panel = build_panel(ps, k_steps=K_STEPS, h_steps=H_STEPS)
        if panel is None or len(panel) < 200:
            continue

        panel["market_id"] = mid
        panel["liq"] = float(m.get("liquidity") or 0)
        panels.append(panel)
        kept += 1

    if not panels:
        print("No usable markets after pulling trades from subgraph.")
        print("Most likely: subgraph URL/schema mismatch. Paste the first error message and I’ll adjust.")
        return

    data = pd.concat(panels, ignore_index=True)
    print("\nUsable markets:", kept)
    print("Total obs:", len(data))

    # 3) Baseline FE + cluster
    fe = demean_by_group(data, "market_id", ["dph", "dp0"])
    m1 = ols_cluster(fe["dph"], fe[["dp0"]], groups=data["market_id"])
    print("\n=== FE + Cluster(market): dph ~ dp0 ===")
    print(m1.summary())


if __name__ == "__main__":
    main()