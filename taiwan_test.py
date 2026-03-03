import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm

BASE = "https://api.manifold.markets/v0"

# 1) Get market by slug (slug is the URL part after the username)
# From the market page URL:
# https://manifold.markets/BTE/will-lai-chingte-be-elected-preside
MARKET_SLUG = "will-lai-chingte-be-elected-preside"

def get_market_by_slug(slug: str) -> dict:
    r = requests.get(f"{BASE}/slug/{slug}", timeout=30)
    r.raise_for_status()
    return r.json()

def get_bets(contract_id: str, max_bets: int = 20000) -> list[dict]:
    """Paginate bets backwards using 'before' cursor (createdTime)."""
    all_bets = []
    before = None
    while True:
        params = {"contractId": contract_id, "limit": 1000}
        if before is not None:
            params["before"] = before
        r = requests.get(f"{BASE}/bets", params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        all_bets.extend(batch)
        before = batch[-1]["createdTime"]
        if len(batch) < 1000 or len(all_bets) >= max_bets:
            break
    return all_bets[:max_bets]

def prob_timeseries_from_bets(bets: list[dict]) -> pd.DataFrame:
    rows = []
    for b in bets:
        if "probAfter" in b and b.get("createdTime") is not None:
            rows.append((b["createdTime"], b["probAfter"], b.get("amount", np.nan)))
    df = pd.DataFrame(rows, columns=["t_ms", "p", "amount"])
    df = df.sort_values("t_ms").drop_duplicates("t_ms")
    df["t"] = pd.to_datetime(df["t_ms"], unit="ms", utc=True)
    return df[["t", "p", "amount"]].reset_index(drop=True)

def resample_probs(df: pd.DataFrame, freq: str = "6H") -> pd.DataFrame:
    # last observation carried forward
    s = df.set_index("t")["p"].resample(freq).last().ffill()
    out = s.reset_index().rename(columns={"p": "p"})
    return out

def build_spike_panel(prob_df: pd.DataFrame, k_steps: int, h_steps: int) -> pd.DataFrame:
    """
    prob_df must be regularly sampled.
    k_steps: spike window in steps
    h_steps: reversal horizon in steps
    """
    p = prob_df["p"].values
    t = prob_df["t"].values

    idx = np.arange(k_steps, len(p) - h_steps)
    dp0 = p[idx] - p[idx - k_steps]
    dph = p[idx + h_steps] - p[idx]

    return pd.DataFrame({
        "t": t[idx],
        "p_t": p[idx],
        "dp0": dp0,
        "dph": dph,
        "abs_dp0": np.abs(dp0),
    })

def run_reversal_regression(panel: pd.DataFrame):
    X = sm.add_constant(panel["dp0"])
    y = panel["dph"]
    return sm.OLS(y, X).fit(cov_type="HC3")

# ---- Run it ----
mkt = get_market_by_slug(MARKET_SLUG)
contract_id = mkt["id"]
print("Market:", mkt["question"])
print("Resolved:", mkt.get("isResolved"), "Resolution:", mkt.get("resolution"))

bets = get_bets(contract_id)
ts = prob_timeseries_from_bets(bets)

# Resample to 6-hour bins
rs = resample_probs(ts, freq="6h")

# Define spike window and reversal horizon:
# k_steps=1 => 6h spike, h_steps=2 => 12h reversal
panel = build_spike_panel(rs, k_steps=1, h_steps=2)

# Optional: focus on big spikes only (top 10% absolute moves)
threshold = panel["abs_dp0"].quantile(0.90)
big = panel[panel["abs_dp0"] > 0.10]   # 10pp move

print("Total observations:", len(panel), "Big-spike observations:", len(big))

model_all = run_reversal_regression(panel)
model_big = run_reversal_regression(big)

print("\nReversal regression (all): dph ~ dp0")
print(model_all.summary())

print("\nReversal regression (big spikes): dph ~ dp0")
print(model_big.summary())
