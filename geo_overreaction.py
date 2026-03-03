import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm

BASE = "https://api.manifold.markets/v0"

# -----------------------------
# Config you can tweak
# -----------------------------
KEYWORDS = [
    "Gaza ceasefire",
    "Israel ceasefire",
    "hostage deal",
    "Israel Hamas",
    "Ukraine ceasefire",
    "Ukraine escalation",
    "Russia Ukraine",
    "invasion",
    "sanctions",
]

MAX_MARKETS_PER_KEYWORD = 25     # keep runtime reasonable
MAX_MARKETS_TOTAL = 60           # cap total markets used
MIN_BETS = 80                    # minimum bet count to keep market
MIN_VOLUME = 500                 # market volume filter (Manifold points)
RESAMPLE_FREQ = "1h"             # try "1h" or "2h" (lowercase required)
K_STEPS = 3                      # spike window in steps (3 * 1h = 3h)
H_STEPS = 12                     # reversal horizon (12 * 1h = 12h)
BIG_SPIKE_ABS = 0.05             # 5 percentage-point move threshold
MAX_BETS_PER_MARKET = 20000      # cap pagination

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
    """
    Fetch bets for a market using time-based pagination.
    Uses GET /v0/bets with afterTime + order=asc (supported by Manifold).
    """
    all_bets = []
    after_time = 0  # ms since epoch
    last_after_time = None

    while True:
        params = {
            "contractId": contract_id,
            "limit": 1000,
            "order": "asc",
            "afterTime": after_time,
        }

        r = requests.get(f"{BASE}/bets", params=params, timeout=30)

        # Some markets may not expose bets; skip them
        if r.status_code in (404, 400):
            return []

        r.raise_for_status()
        batch = r.json()
        if not batch:
            break

        all_bets.extend(batch)

        # advance cursor using createdTime of last returned bet
        after_time = batch[-1]["createdTime"]

        # safety: if cursor doesn't advance, stop (prevents infinite loop)
        if last_after_time == after_time:
            break
        last_after_time = after_time

        if len(batch) < 1000 or len(all_bets) >= max_bets:
            break

        # move 1ms forward to avoid refetching the last bet if API treats afterTime as inclusive
        after_time += 1

    return all_bets[:max_bets]

# -----------------------------
# Transformations
# -----------------------------
def prob_timeseries_from_bets(bets):
    rows = []
    for b in bets:
        if "probAfter" in b and b.get("createdTime") is not None:
            rows.append((b["createdTime"], b["probAfter"], b.get("amount", np.nan)))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["t_ms", "p", "amount"])
    df = df.sort_values("t_ms").drop_duplicates("t_ms")
    df["t"] = pd.to_datetime(df["t_ms"], unit="ms", utc=True)
    return df[["t", "p", "amount"]].reset_index(drop=True)

def resample_probs(ts, freq="1h"):
    s = ts.set_index("t")["p"].resample(freq).last().ffill()
    return s.reset_index().rename(columns={"p": "p"})

def build_panel(prob_df, k_steps, h_steps):
    p = prob_df["p"].values
    t = prob_df["t"].values
    if len(p) <= (k_steps + h_steps + 5):
        return None
    idx = np.arange(k_steps, len(p) - h_steps)
    dp0 = p[idx] - p[idx - k_steps]
    dph = p[idx + h_steps] - p[idx]
    out = pd.DataFrame({
        "t": t[idx],
        "p_t": p[idx],
        "dp0": dp0,
        "dph": dph,
        "abs_dp0": np.abs(dp0),
    })
    return out

def demean_by_group(df, group_col, cols):
    # within transformation for fixed effects
    g = df.groupby(group_col)[cols].transform("mean")
    out = df.copy()
    for c in cols:
        out[c] = out[c] - g[c]
    return out

# -----------------------------
# Regression runners
# -----------------------------
def ols_hc3(y, X):
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, Xc).fit(cov_type="HC3")

def main():
    # 1) collect candidate markets
    market_rows = []
    for kw in KEYWORDS:
        ms = search_markets(kw, limit=MAX_MARKETS_PER_KEYWORD)
        for m in ms:
            market_rows.append({
                "keyword": kw,
                "id": m["id"],
                "question": m.get("question"),
                "url": m.get("url"),
                "volume": float(m.get("volume") or 0),
                "liquidity": float(m.get("liquidity") or 0),
                "closeTime": m.get("closeTime"),
                "createdTime": m.get("createdTime"),
                "isResolved": m.get("isResolved"),
                "resolution": m.get("resolution"),
                "resolutionTime": m.get("resolutionTime"),
            })

    markets = pd.DataFrame(market_rows).drop_duplicates("id")
    markets = markets[(markets["isResolved"] == True)]
    markets = markets[markets["volume"] >= MIN_VOLUME].copy()
    markets = markets.sort_values(["liquidity", "volume"], ascending=False).head(MAX_MARKETS_TOTAL)

    print(f"Candidate markets after filters: {len(markets)}")
    if len(markets) == 0:
        print("No markets found with current filters. Lower MIN_VOLUME or increase MAX_MARKETS_PER_KEYWORD.")
        return

    # 2) download + build panel
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

        panel["market_id"] = m["id"]
        panel["question"] = m["question"]
        panel["keyword"] = m["keyword"]
        panel["volume"] = m["volume"]
        panel["liquidity"] = m["liquidity"]
        panel["n_bets"] = len(bets)

        panels.append(panel)
        kept += 1

    if not panels:
        print("No usable markets after building panels. Try lowering MIN_BETS / MIN_VOLUME or using RESAMPLE_FREQ='2h'.")
        return

    data = pd.concat(panels, ignore_index=True)

    print("\nUsable markets:", kept)
    print("Total panel observations:", len(data))
    print("Markets with most obs:\n", data["market_id"].value_counts().head(5))

    # 3) big spikes subset
    big = data[data["abs_dp0"] >= BIG_SPIKE_ABS].copy()
    print(f"\nBig spike threshold: abs(dp0) >= {BIG_SPIKE_ABS:.2f}")
    print("Big spike observations:", len(big))
    print("Markets contributing big spikes:", big["market_id"].nunique())

    # 4) Liquidity indicator (low liquidity = bottom tercile)
    q1 = data["liquidity"].quantile(0.33)
    data["low_liq"] = (data["liquidity"] <= q1).astype(int)
    data["dp0_x_lowliq"] = data["dp0"] * data["low_liq"]

    # -----------------------------
    # Regressions
    # -----------------------------
    print("\n=== Regression 1: Pooled (all obs) dph ~ dp0 ===")
    m1 = ols_hc3(data["dph"], data[["dp0"]])
    print(m1.summary())

    print("\n=== Regression 2: Fixed effects (demeaned by market) dph ~ dp0 ===")
    fe = demean_by_group(data, "market_id", ["dph", "dp0"])
    m2 = ols_hc3(fe["dph"], fe[["dp0"]])
    print(m2.summary())

    if len(big) >= 50:
        print("\n=== Regression 3: Big spikes only dph ~ dp0 ===")
        m3 = ols_hc3(big["dph"], big[["dp0"]])
        print(m3.summary())
    else:
        print("\n=== Regression 3 skipped: not enough big-spike observations (need ~50+) ===")

    print("\n=== Regression 4: Liquidity interaction dph ~ dp0 + dp0*low_liq + low_liq ===")
    m4 = ols_hc3(data["dph"], data[["dp0", "dp0_x_lowliq", "low_liq"]])
    print(m4.summary())

    # Quick sanity: mean reversal by sign of spike
    data["spike_sign"] = np.sign(data["dp0"])
    bysign = data.groupby("spike_sign")["dph"].mean()
    print("\nMean future change (dph) by spike sign:")
    print(bysign)

if __name__ == "__main__":
    main()