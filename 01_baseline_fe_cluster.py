import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm

BASE = "https://api.manifold.markets/v0"

# -----------------------------
# Config
# -----------------------------
KEYWORDS = [
    "Gaza ceasefire", "Israel ceasefire", "hostage deal", "Israel Hamas",
    "Ukraine ceasefire", "Ukraine escalation", "Russia Ukraine", "invasion", "sanctions"
]
MAX_MARKETS_PER_KEYWORD = 25
MAX_MARKETS_TOTAL = 60

MIN_VOLUME = 200
MIN_BETS = 40

RESAMPLE_FREQ = "1h"   # lowercase for pandas >=2
K_STEPS = 3            # spike window = 3h
H_STEPS = 12           # reversal horizon = 12h

EXCLUDE_LAST_HOURS_BEFORE_RESOLUTION = 48
MAX_BETS_PER_MARKET = 20000


# -----------------------------
# API helpers
# -----------------------------
def search_markets(term, limit=50):
    url = f"{BASE}/search-markets"
    params = {
        "term": term,
        "sort": "liquidity",
        "filter": "resolved",
        "contractType": "BINARY",
        "limit": limit
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def get_bets(contract_id, max_bets=20000):
    """Time-based pagination using afterTime + order=asc."""
    all_bets = []
    after_time = 0
    last_after = None

    while True:
        params = {
            "contractId": contract_id,
            "limit": 1000,
            "order": "asc",
            "afterTime": after_time
        }
        r = requests.get(f"{BASE}/bets", params=params, timeout=30)

        # Some markets can error; skip cleanly
        if r.status_code in (400, 404):
            return []

        r.raise_for_status()
        batch = r.json()
        if not batch:
            break

        all_bets.extend(batch)
        after_time = batch[-1]["createdTime"]

        if last_after == after_time:
            break
        last_after = after_time

        if len(batch) < 1000 or len(all_bets) >= max_bets:
            break

        after_time += 1  # avoid refetch if inclusive

    return all_bets[:max_bets]


# -----------------------------
# Data transforms
# -----------------------------
def prob_timeseries_from_bets(bets):
    rows = []
    for b in bets:
        if "probAfter" in b and b.get("createdTime") is not None:
            rows.append((b["createdTime"], b["probAfter"]))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["t_ms", "p"]).sort_values("t_ms").drop_duplicates("t_ms")
    df["t"] = pd.to_datetime(df["t_ms"], unit="ms", utc=True)
    return df[["t", "p"]].reset_index(drop=True)

def resample_probs(ts, freq="1h"):
    s = ts.set_index("t")["p"].resample(freq).last().ffill()
    out = s.reset_index().rename(columns={"p": "p"})
    # Ensure tz-aware UTC timestamps
    out["t"] = pd.to_datetime(out["t"], utc=True)
    return out

def build_panel(prob_df, k_steps, h_steps):
    p = prob_df["p"].to_numpy()
    # IMPORTANT: keep t as pandas Series to preserve tz-awareness
    t = prob_df["t"].reset_index(drop=True)

    if len(p) <= (k_steps + h_steps + 5):
        return None

    idx = np.arange(k_steps, len(p) - h_steps)
    dp0 = p[idx] - p[idx - k_steps]
    dph = p[idx + h_steps] - p[idx]

    out = pd.DataFrame({
        "t": t.iloc[idx].to_numpy(),   # stays tz-aware when coming from tz-aware Series
        "p_t": p[idx],
        "dp0": dp0,
        "dph": dph,
        "abs_dp0": np.abs(dp0),
    })
    return out

def demean_by_group(df, group_col, cols):
    g = df.groupby(group_col)[cols].transform("mean")
    out = df.copy()
    for c in cols:
        out[c] = out[c] - g[c]
    return out


# -----------------------------
# Econometrics
# -----------------------------
def ols_cluster(y, X, groups):
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": groups})


def main():
    # 1) collect candidate markets
    rows = []
    for kw in KEYWORDS:
        for m in search_markets(kw, limit=MAX_MARKETS_PER_KEYWORD):
            rows.append({
                "keyword": kw,
                "id": m["id"],
                "question": m.get("question"),
                "url": m.get("url"),
                "volume": float(m.get("volume") or 0),
                "liquidity": float(m.get("liquidity") or 0),
                "isResolved": m.get("isResolved"),
                "resolutionTime": m.get("resolutionTime"),
            })

    markets = pd.DataFrame(rows).drop_duplicates("id")
    markets = markets[markets["isResolved"] == True]
    markets = markets[markets["volume"] >= MIN_VOLUME].copy()
    markets = markets.sort_values(["liquidity", "volume"], ascending=False).head(MAX_MARKETS_TOTAL)

    print(f"Candidate markets after filters: {len(markets)}")
    if len(markets) == 0:
        print("No markets found. Lower MIN_VOLUME or change KEYWORDS.")
        return

    # 2) build panel for each market
    panels = []
    kept = 0

    for _, m in tqdm(markets.iterrows(), total=len(markets)):
        bets = get_bets(m["id"], max_bets=MAX_BETS_PER_MARKET)
        if len(bets) < MIN_BETS:
            continue

        ts = prob_timeseries_from_bets(bets)
        if ts is None or len(ts) < 30:
            continue

        rs = resample_probs(ts, freq=RESAMPLE_FREQ)
        panel = build_panel(rs, k_steps=K_STEPS, h_steps=H_STEPS)
        if panel is None or len(panel) < 50:
            continue

        # exclude last X hours before resolution (avoid mechanical convergence)
        if pd.notnull(m["resolutionTime"]):
            res_t = pd.to_datetime(int(m["resolutionTime"]), unit="ms", utc=True)
            cutoff = res_t - pd.Timedelta(hours=EXCLUDE_LAST_HOURS_BEFORE_RESOLUTION)
            panel = panel[panel["t"] < cutoff]

        if len(panel) < 50:
            continue

        panel["market_id"] = m["id"]
        panel["question"] = m["question"]
        panel["keyword"] = m["keyword"]
        panel["volume"] = m["volume"]
        panel["liquidity"] = m["liquidity"]
        panels.append(panel)
        kept += 1

    if not panels:
        print("No usable markets. Lower MIN_BETS/MIN_VOLUME or use RESAMPLE_FREQ='2h'.")
        return

    data = pd.concat(panels, ignore_index=True)

    print("\nUsable markets:", kept)
    print("Total panel observations:", len(data))
    print("Markets with most obs:\n", data["market_id"].value_counts().head(5))

    # 3) FE (within) regression with clustered SE
    fe = demean_by_group(data, "market_id", ["dph", "dp0"])
    m_fe = ols_cluster(fe["dph"], fe[["dp0"]], groups=data["market_id"])

    print("\n=== FE + Cluster(market) : dph ~ dp0 ===")
    print(m_fe.summary())

    # quick sanity: mean future change by spike sign
    data["spike_sign"] = np.sign(data["dp0"])
    print("\nMean future change (dph) by spike sign:")
    print(data.groupby("spike_sign")["dph"].mean())


if __name__ == "__main__":
    main()