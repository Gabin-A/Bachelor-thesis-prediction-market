import json
import time
import math
import requests
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

# -----------------------------
# Endpoints (public, no auth)
# -----------------------------
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

# -----------------------------
# Config
# -----------------------------
KEYWORDS = [
    "Taiwan election", "China Taiwan", "Taiwan invasion",
    "Gaza ceasefire", "Israel ceasefire", "hostage deal",
    "Ukraine ceasefire", "Russia Ukraine", "sanctions",
    "NATO", "Iran", "Houthi"
]

# Search paging (Gamma public-search uses page, limit_per_type)
LIMIT_PER_TYPE = 50
MAX_PAGES = 3          # per keyword
MAX_MARKETS_TOTAL = 80 # cap overall to keep runtime reasonable

# Filters
MIN_VOLUME_USD = 50_000      # Polymarket volumeNum is often numeric (USD). Adjust if too strict.
REQUIRE_ORDERBOOK = True     # only markets tradable on CLOB
KEEP_CLOSED = 1              # include closed markets in search results

# Frequency & windows
# We request history at interval=1h, but may fall back if too sparse.
HISTORY_INTERVALS = ["1h", "6h", "1d"]  # fast mode fallbacks
K_STEPS = 3       # dp0 window (3 * interval)
H_STEPS = 12      # dph horizon (12 * interval)
MIN_POINTS = 150  # minimum history points to be usable at a given interval

# Preprocessing
EXCLUDE_LAST_HOURS_BEFORE_ENDDATE = 48  # similar to your "exclude last 48h" idea
REQUEST_SLEEP = 0.10                   # be nice to API

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

def gamma_public_search(q: str, page: int = 1):
    """
    Gamma public search:
    GET https://gamma-api.polymarket.com/public-search?q=...&page=...&limit_per_type=...&keep_closed_markets=1
    """
    params = {
        "q": q,
        "page": page,
        "limit_per_type": LIMIT_PER_TYPE,
        "keep_closed_markets": KEEP_CLOSED,
        "search_tags": False,
        "search_profiles": False,
        "optimized": True,
    }
    r = requests.get(f"{GAMMA}/public-search", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def parse_clob_token_ids(market_obj):
    """
    Gamma market objects often contain clobTokenIds as a string (JSON list).
    We need the YES token id (usually index 0 corresponds to 'Yes' in outcomes).
    """
    token_ids_raw = market_obj.get("clobTokenIds")
    outcomes_raw = market_obj.get("outcomes")

    if token_ids_raw is None:
        return None, None

    # token_ids_raw may already be a list or a JSON string
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

    # Prefer explicit "Yes" outcome if outcomes are available
    yes_idx = 0
    if isinstance(outcomes, list):
        for i, o in enumerate(outcomes):
            if isinstance(o, str) and o.strip().lower() == "yes":
                yes_idx = i
                break

    yes_token = str(token_ids[yes_idx])
    no_token = str(token_ids[1 - yes_idx]) if len(token_ids) >= 2 else None
    return yes_token, no_token

def clob_prices_history(token_id: str, interval: str):
    """
    CLOB prices history:
    GET https://clob.polymarket.com/prices-history?market=<token_id>&interval=1h
    Returns: {"history":[{"t":..., "p":...}, ...]}
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
    # t is unix timestamp (seconds) according to docs; sometimes ms appears in practice
    # we detect by magnitude
    if "t" not in df.columns or "p" not in df.columns:
        return None
    t = df["t"].astype("int64")
    unit = "s" if t.median() < 10_000_000_000 else "ms"
    df["t"] = pd.to_datetime(df["t"], unit=unit, utc=True)
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df = df.dropna(subset=["p"]).sort_values("t")
    df = df.drop_duplicates(subset=["t"])
    return df[["t", "p"]].reset_index(drop=True)

def build_panel(prob_df: pd.DataFrame, k_steps: int, h_steps: int):
    """
    Same design as before:
    dp0_t = p_t - p_{t-k}
    dph_t = p_{t+h} - p_t
    """
    p = prob_df["p"].to_numpy()
    t = prob_df["t"].reset_index(drop=True)  # keep tz-aware

    if len(p) <= (k_steps + h_steps + 5):
        return None

    idx = np.arange(k_steps, len(p) - h_steps)
    dp0 = p[idx] - p[idx - k_steps]
    dph = p[idx + h_steps] - p[idx]

    out = pd.DataFrame({
        "t": t.iloc[idx].to_numpy(),
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

def ols_cluster(y, X, groups):
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": groups})

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    # 1) Discover markets via Gamma public-search
    found = []
    seen_ids = set()

    for kw in KEYWORDS:
        for page in range(1, MAX_PAGES + 1):
            js = gamma_public_search(kw, page=page)
            events = js.get("events") or []
            # Each event has "markets" list in many responses
            for ev in events:
                markets = ev.get("markets") or []
                for m in markets:
                    mid = m.get("id")
                    if not mid or mid in seen_ids:
                        continue

                    # Basic filters
                    enable_ob = bool(m.get("enableOrderBook", False))
                    if REQUIRE_ORDERBOOK and not enable_ob:
                        continue

                    vol = safe_float(m.get("volumeNum"), safe_float(m.get("volume")))
                    liq = safe_float(m.get("liquidityNum"), safe_float(m.get("liquidity")))

                    # keep markets with decent volume
                    if vol < MIN_VOLUME_USD:
                        continue

                    yes_token, _ = parse_clob_token_ids(m)
                    if yes_token is None:
                        continue

                    found.append({
                        "market_id": str(mid),
                        "question": m.get("question", ""),
                        "slug": m.get("slug", ""),
                        "endDate": m.get("endDateIso") or m.get("endDate"),
                        "closed": bool(m.get("closed", False)),
                        "active": bool(m.get("active", False)),
                        "volume": vol,
                        "liquidity": liq,
                        "yes_token": yes_token,
                    })
                    seen_ids.add(mid)

                    if len(found) >= MAX_MARKETS_TOTAL:
                        break
                if len(found) >= MAX_MARKETS_TOTAL:
                    break
            if len(found) >= MAX_MARKETS_TOTAL:
                break
            time.sleep(REQUEST_SLEEP)

        if len(found) >= MAX_MARKETS_TOTAL:
            break

    markets = pd.DataFrame(found).drop_duplicates("market_id")
    markets = markets.sort_values(["liquidity", "volume"], ascending=False)
    print(f"Candidate markets after filters: {len(markets)}")
    if len(markets) == 0:
        print("No markets found. Try lowering MIN_VOLUME_USD or changing KEYWORDS.")
        return

    # 2) Pull price history and build panel
    panels = []
    kept = 0

    for _, m in tqdm(markets.iterrows(), total=len(markets)):
        token = m["yes_token"]

        prob_df = None
        used_interval = None

        # try intervals from fine to coarse
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

        # Exclude last N hours before endDate (if available) to avoid mechanical end effects
        if m["endDate"]:
            try:
                end_t = pd.to_datetime(m["endDate"], utc=True, errors="coerce")
                if pd.notnull(end_t):
                    cutoff = end_t - pd.Timedelta(hours=EXCLUDE_LAST_HOURS_BEFORE_ENDDATE)
                    panel = panel[panel["t"] < cutoff]
            except Exception:
                pass

        if len(panel) < 50:
            continue

        # fast attention proxy: how dense the history points are around t (not perfect)
        # We'll approximate attention by "points per hour" in the last k-window:
        # If interval is 1h, this is basically 1; if interval is coarser, it changes.
        interval_minutes = {"1h": 60, "6h": 360, "1d": 1440}.get(used_interval, 60)
        panel["attn_proxy"] = 60 / interval_minutes  # higher means more frequent points

        panel["market_id"] = m["market_id"]
        panel["question"] = m["question"]
        panel["liquidity"] = float(m["liquidity"])
        panel["volume"] = float(m["volume"])
        panel["interval_used"] = used_interval

        panels.append(panel)
        kept += 1

        time.sleep(REQUEST_SLEEP)

    if not panels:
        print("No usable markets after pulling history. This can happen if /prices-history is too coarse for your targets.")
        print("Try increasing HISTORY_INTERVALS to include 'all' or lowering MIN_POINTS.")
        return

    data = pd.concat(panels, ignore_index=True)

    print("\nUsable markets:", kept)
    print("Total panel observations:", len(data))
    print("Intervals used:\n", data["interval_used"].value_counts())

    # -----------------------------
    # 3) Baseline FE + cluster
    # -----------------------------
    fe_base = demean_by_group(data, "market_id", ["dph", "dp0"])
    m1 = ols_cluster(fe_base["dph"], fe_base[["dp0"]], groups=data["market_id"])
    print("\n=== (1) FE + Cluster(market): dph ~ dp0 ===")
    print(m1.summary())

    # -----------------------------
    # 4) Liquidity mechanism
    # -----------------------------
    data["log_liq"] = np.log(np.clip(data["liquidity"], 1e-9, None))
    data["dp0_x_logliq"] = data["dp0"] * data["log_liq"]

    fe_liq = demean_by_group(data, "market_id", ["dph", "dp0", "dp0_x_logliq"])
    m2 = ols_cluster(fe_liq["dph"], fe_liq[["dp0", "dp0_x_logliq"]], groups=data["market_id"])
    print("\n=== (2) FE + Cluster(market): dph ~ dp0 + dp0×log(liquidity) ===")
    print(m2.summary())

    # -----------------------------
    # 5) Attention mechanism (FAST proxy)
    # -----------------------------
    data["log_attn"] = np.log1p(data["attn_proxy"])
    data["dp0_x_attn"] = data["dp0"] * data["log_attn"]

    fe_attn = demean_by_group(data, "market_id", ["dph", "dp0", "dp0_x_attn"])
    m3 = ols_cluster(fe_attn["dph"], fe_attn[["dp0", "dp0_x_attn"]], groups=data["market_id"])
    print("\n=== (3) FE + Cluster(market): dph ~ dp0 + dp0×log(1+attn_proxy) [FAST] ===")
    print(m3.summary())

    # small sanity prints
    data["spike_sign"] = np.sign(data["dp0"])
    print("\nMean future change (dph) by spike sign:")
    print(data.groupby("spike_sign")["dph"].mean())

    # save dataset for later (research-grade upgrade)
    out_csv = "polymarket_fast_panel.csv"
    data.to_csv(out_csv, index=False)
    print(f"\nSaved panel to: {out_csv}")


if __name__ == "__main__":
    main()