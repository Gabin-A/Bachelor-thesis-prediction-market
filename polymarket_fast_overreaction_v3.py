import json
import time
import requests
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

# -----------------------------
# Config
# -----------------------------
KEYWORDS = [
    "taiwan", "china taiwan", "gaza", "ceasefire", "israel", "hostage",
    "ukraine", "russia", "sanctions", "iran", "nato", 'election', 'war', 'ceasfire'
]

# How many markets to scan from Gamma
GAMMA_LIMIT = 200       # per request
GAMMA_MAX_PAGES = 20    # total scanned = limit * pages (e.g., 4,000)

# After keyword filtering, cap markets to pull history for (speed)
MAX_MARKETS_TOTAL = 80

# Filters (start loose, tighten later)
REQUIRE_ORDERBOOK = False
MIN_VOLUME_USD = 0       # try 0 if still too strict
MIN_LIQUIDITY = 0           # try 0 always ok

# Price-history intervals to try (fine -> coarse)
HISTORY_INTERVALS = ["6h", "1d"]
MIN_POINTS = 30

# Windows in "steps" of the chosen interval
K_STEPS = 3
H_STEPS = 12

# End effects
EXCLUDE_LAST_HOURS_BEFORE_ENDDATE = 48

REQUEST_SLEEP = 0.10


# -----------------------------
# Helpers
# -----------------------------
def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def gamma_list_markets(limit=200, offset=0):
    """
    GET /markets (Gamma API)
    """
    params = {"limit": limit, "offset": offset}
    r = requests.get(f"{GAMMA}/markets", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def parse_clob_token_ids(market_obj):
    """
    clobTokenIds is often a JSON string list: ["YES_TOKEN_ID","NO_TOKEN_ID"]
    outcomes may be JSON string: ["Yes","No"]
    """
    token_ids_raw = market_obj.get("clobTokenIds")
    outcomes_raw = market_obj.get("outcomes")

    if token_ids_raw is None:
        return None, None

    if isinstance(token_ids_raw, str):
        try:
            token_ids = json.loads(token_ids_raw)
        except Exception:
            return None, None
    else:
        token_ids = token_ids_raw

    if isinstance(outcomes_raw, str):
        try:
            outcomes = json.loads(outcomes_raw)
        except Exception:
            outcomes = None
    else:
        outcomes = outcomes_raw

    if not isinstance(token_ids, list) or len(token_ids) < 2:
        return None, None

    yes_idx = 0
    if isinstance(outcomes, list):
        for i, o in enumerate(outcomes):
            if isinstance(o, str) and o.strip().lower() == "yes":
                yes_idx = i
                break

    yes_token = str(token_ids[yes_idx])
    no_token = str(token_ids[1 - yes_idx])
    return yes_token, no_token

def clob_prices_history(token_id: str, interval: str):
    """
    GET /prices-history (CLOB)
    params: market=<token_id>&interval=<interval>
    """
    params = {"market": token_id, "interval": interval}
    r = requests.get(f"{CLOB}/prices-history", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def history_to_df(hist_json):
    hist = hist_json.get("history", []) or []
    if not hist:
        return None

    df = pd.DataFrame(hist)
    if "t" not in df.columns or "p" not in df.columns:
        return None

    # detect seconds vs ms
    t_num = pd.to_numeric(df["t"], errors="coerce").dropna()
    if len(t_num) == 0:
        return None

    t_med = float(t_num.median())
    unit = "s" if t_med < 10_000_000_000 else "ms"

    df["t"] = pd.to_datetime(df["t"], unit=unit, utc=True, errors="coerce")
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df = df.dropna(subset=["t", "p"]).sort_values("t").drop_duplicates(subset=["t"])
    return df[["t", "p"]].reset_index(drop=True)

def build_panel(prob_df: pd.DataFrame, k_steps: int, h_steps: int):
    p = prob_df["p"].to_numpy()
    t = prob_df["t"].reset_index(drop=True)  # tz-aware preserved

    if len(p) <= (k_steps + h_steps + 5):
        return None

    idx = np.arange(k_steps, len(p) - h_steps)
    dp0 = p[idx] - p[idx - k_steps]
    dph = p[idx + h_steps] - p[idx]

    return pd.DataFrame({
        "t": t.iloc[idx].to_numpy(),
        "p_t": p[idx],
        "dp0": dp0,
        "dph": dph,
        "abs_dp0": np.abs(dp0),
    })

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
# Main
# -----------------------------
def main():
    # 1) Pull markets from Gamma
    all_markets = []
    for page in range(GAMMA_MAX_PAGES):
        offset = page * GAMMA_LIMIT
        batch = gamma_list_markets(limit=GAMMA_LIMIT, offset=offset)
        if not isinstance(batch, list) or len(batch) == 0:
            break
        all_markets.extend(batch)
        time.sleep(REQUEST_SLEEP)

    print("Total markets pulled from /markets:", len(all_markets))
    if len(all_markets) == 0:
        print("Gamma returned 0 markets (unexpected). Check connectivity.")
        return

    # 2) Filter locally by keyword in question
    kw_set = [k.lower() for k in KEYWORDS]
    selected = []
    for m in all_markets:
        q = (m.get("question") or "").lower()
        if not q:
            continue
        if not any(k in q for k in kw_set):
            continue

        # filters
        if REQUIRE_ORDERBOOK and not bool(m.get("enableOrderBook", False)):
            continue

        vol = safe_float(m.get("volumeNum"), safe_float(m.get("volume"), 0.0))
        liq = safe_float(m.get("liquidityNum"), safe_float(m.get("liquidity"), 0.0))

        if vol < MIN_VOLUME_USD or liq < MIN_LIQUIDITY:
            continue

        yes_token, _ = parse_clob_token_ids(m)
        if yes_token is None:
            continue

        selected.append({
            "market_id": str(m.get("id")),
            "question": m.get("question", ""),
            "endDate": m.get("endDateIso") or m.get("endDate"),
            "volume": vol,
            "liquidity": liq,
            "yes_token": yes_token,
            "closed": bool(m.get("closed", False)),
            "active": bool(m.get("active", False)),
        })

    markets = pd.DataFrame(selected).drop_duplicates("market_id")
    print("Keyword-matched markets after filters:", len(markets))

    if len(markets) == 0:
        print("\nNo matches. Quick fixes:")
        print("- Set REQUIRE_ORDERBOOK=False")
        print("- Set MIN_VOLUME_USD=0")
        print("- Add broader keywords (e.g., 'election', 'war', 'ceasefire')")
        return

    markets = markets.sort_values(["liquidity", "volume"], ascending=False).head(MAX_MARKETS_TOTAL).reset_index(drop=True)
    print("Markets to download history for:", len(markets))

    # 3) Pull price history, build panel
    panels = []
    kept = 0

    for _, m in tqdm(markets.iterrows(), total=len(markets)):
        token = m["yes_token"]

        prob_df = None
        used_interval = None

        for interval in HISTORY_INTERVALS:
            try:
                hist = clob_prices_history(token, interval=interval)
                df = history_to_df(hist)
                if df is None or len(df) < MIN_POINTS:
                    continue
                prob_df = df
                used_interval = interval
                break
            except Exception:
                continue

        if prob_df is None:
            continue

        panel = build_panel(prob_df, k_steps=K_STEPS, h_steps=H_STEPS)
        if panel is None or len(panel) < 50:
            continue

        # exclude last X hours before endDate (if parseable)
        if m.get("endDate"):
            end_t = pd.to_datetime(m["endDate"], utc=True, errors="coerce")
            if pd.notnull(end_t):
                cutoff = end_t - pd.Timedelta(hours=EXCLUDE_LAST_HOURS_BEFORE_ENDDATE)
                panel = panel[panel["t"] < cutoff]

        if len(panel) < 50:
            continue

        panel["market_id"] = m["market_id"]
        panel["question"] = m["question"]
        panel["liquidity"] = float(m["liquidity"])
        panel["volume"] = float(m["volume"])
        panel["interval_used"] = used_interval
        panels.append(panel)
        kept += 1
        time.sleep(REQUEST_SLEEP)

    if not panels:
        print("\nNo usable markets after /prices-history.")
        print("This often means /prices-history is too coarse for these markets.")
        print("Try: MIN_POINTS=30, or use HISTORY_INTERVALS=['6h','1d'], or switch to research-grade (subgraph).")
        return

    data = pd.concat(panels, ignore_index=True)
    print("\nUsable markets:", kept)
    print("Total panel observations:", len(data))
    print("Intervals used:\n", data["interval_used"].value_counts())

    # (1) Baseline FE + cluster
    fe_base = demean_by_group(data, "market_id", ["dph", "dp0"])
    m1 = ols_cluster(fe_base["dph"], fe_base[["dp0"]], groups=data["market_id"])
    print("\n=== (1) FE + Cluster(market): dph ~ dp0 ===")
    print(m1.summary())

    # (2) Liquidity mechanism
    data["log_liq"] = np.log(np.clip(data["liquidity"], 1e-9, None))
    data["dp0_x_logliq"] = data["dp0"] * data["log_liq"]
    fe_liq = demean_by_group(data, "market_id", ["dph", "dp0", "dp0_x_logliq"])
    m2 = ols_cluster(fe_liq["dph"], fe_liq[["dp0", "dp0_x_logliq"]], groups=data["market_id"])
    print("\n=== (2) FE + Cluster(market): dph ~ dp0 + dp0×log(liquidity) ===")
    print(m2.summary())

    # sanity
    data["spike_sign"] = np.sign(data["dp0"])
    print("\nMean future change (dph) by spike sign:")
    print(data.groupby("spike_sign")["dph"].mean())

    out_csv = "polymarket_fast_panel.csv"
    data.to_csv(out_csv, index=False)
    print(f"\nSaved panel to: {out_csv}")


if __name__ == "__main__":
    main()