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
# Config (FAST)
# -----------------------------
KEYWORDS = [
    "Taiwan", "China Taiwan",
    "Gaza ceasefire", "Israel ceasefire", "hostage",
    "Ukraine ceasefire", "Russia Ukraine", "sanctions",
    "NATO", "Iran"
]

LIMIT_PER_TYPE = 50
MAX_PAGES = 3
MAX_MARKETS_TOTAL = 80

# Relaxed filters (you can raise later)
MIN_VOLUME_USD = 0          # <-- start at 0 to ensure you get data
REQUIRE_ORDERBOOK = False   # <-- start False; later set True if needed
KEEP_CLOSED = 1

HISTORY_INTERVALS = ["1h", "6h", "1d"]
K_STEPS = 3
H_STEPS = 12
MIN_POINTS = 80             # <-- start lower; later raise (150) if you get good density

EXCLUDE_LAST_HOURS_BEFORE_ENDDATE = 48
REQUEST_SLEEP = 0.10
MAX_BETS_PER_MARKET = 20000  # unused here, kept for symmetry


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def gamma_public_search(q: str, page: int = 1):
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
    no_token = str(token_ids[1 - yes_idx]) if len(token_ids) >= 2 else None
    return yes_token, no_token


def clob_prices_history(token_id: str, interval: str):
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

    t = pd.to_numeric(df["t"], errors="coerce").dropna().astype("int64")
    if len(t) == 0:
        return None
    unit = "s" if t.median() < 10_000_000_000 else "ms"
    df["t"] = pd.to_datetime(df["t"], unit=unit, utc=True, errors="coerce")
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df = df.dropna(subset=["t", "p"]).sort_values("t").drop_duplicates(subset=["t"])
    return df[["t", "p"]].reset_index(drop=True)


def build_panel(prob_df: pd.DataFrame, k_steps: int, h_steps: int):
    p = prob_df["p"].to_numpy()
    t = prob_df["t"].reset_index(drop=True)

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


def main():
    found = []
    seen = set()

    raw_candidates = 0
    dropped_no_token = 0
    dropped_orderbook = 0
    dropped_volume = 0

    for kw in KEYWORDS:
        for page in range(1, MAX_PAGES + 1):
            js = gamma_public_search(kw, page=page)
            events = js.get("events") or []
            for ev in events:
                for m in (ev.get("markets") or []):
                    raw_candidates += 1
                    mid = m.get("id")
                    if not mid or mid in seen:
                        continue

                    enable_ob = bool(m.get("enableOrderBook", False))
                    if REQUIRE_ORDERBOOK and not enable_ob:
                        dropped_orderbook += 1
                        continue

                    vol = safe_float(m.get("volumeNum"), safe_float(m.get("volume")))
                    liq = safe_float(m.get("liquidityNum"), safe_float(m.get("liquidity")))
                    if vol < MIN_VOLUME_USD:
                        dropped_volume += 1
                        continue

                    yes_token, _ = parse_clob_token_ids(m)
                    if yes_token is None:
                        dropped_no_token += 1
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
                        "enableOrderBook": enable_ob,
                    })
                    seen.add(mid)

                    if len(found) >= MAX_MARKETS_TOTAL:
                        break
                if len(found) >= MAX_MARKETS_TOTAL:
                    break
            if len(found) >= MAX_MARKETS_TOTAL:
                break
            time.sleep(REQUEST_SLEEP)
        if len(found) >= MAX_MARKETS_TOTAL:
            break

    print("Raw candidates scanned:", raw_candidates)
    print("Kept markets:", len(found))
    print("Dropped (no token ids):", dropped_no_token)
    print("Dropped (orderbook filter):", dropped_orderbook)
    print("Dropped (volume filter):", dropped_volume)

    if len(found) == 0:
        print("\nNo markets found. Next steps:")
        print("- Set REQUIRE_ORDERBOOK=False (already).")
        print("- Keep MIN_VOLUME_USD=0 (already).")
        print("- Increase MAX_PAGES or change KEYWORDS.")
        return

    markets = pd.DataFrame(found).drop_duplicates("market_id")

    # Ensure columns exist
    for col in ["liquidity", "volume"]:
        if col not in markets.columns:
            markets[col] = 0.0

    markets["liquidity"] = pd.to_numeric(markets["liquidity"], errors="coerce").fillna(0.0)
    markets["volume"] = pd.to_numeric(markets["volume"], errors="coerce").fillna(0.0)

    markets = markets.sort_values(["liquidity", "volume"], ascending=False).reset_index(drop=True)
    print(f"\nCandidate markets after filters: {len(markets)}")

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

        # exclude last X hours before endDate if available
        if m.get("endDate"):
            end_t = pd.to_datetime(m["endDate"], utc=True, errors="coerce")
            if pd.notnull(end_t):
                cutoff = end_t - pd.Timedelta(hours=EXCLUDE_LAST_HOURS_BEFORE_ENDDATE)
                panel = panel[panel["t"] < cutoff]

        if len(panel) < 50:
            continue

        interval_minutes = {"1h": 60, "6h": 360, "1d": 1440}.get(used_interval, 60)
        panel["attn_proxy"] = 60 / interval_minutes

        panel["market_id"] = m["market_id"]
        panel["question"] = m["question"]
        panel["liquidity"] = float(m["liquidity"])
        panel["volume"] = float(m["volume"])
        panel["interval_used"] = used_interval

        panels.append(panel)
        kept += 1

        time.sleep(REQUEST_SLEEP)

    if not panels:
        print("\nNo usable markets after pulling history.")
        print("Likely cause: /prices-history too sparse for these markets.")
        print("Try lowering MIN_POINTS further (e.g., 30) or include '1w' if supported, or switch to research-grade (subgraph).")
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

    # (3) Attention mechanism (FAST proxy)
    data["log_attn"] = np.log1p(data["attn_proxy"])
    data["dp0_x_attn"] = data["dp0"] * data["log_attn"]
    fe_attn = demean_by_group(data, "market_id", ["dph", "dp0", "dp0_x_attn"])
    m3 = ols_cluster(fe_attn["dph"], fe_attn[["dp0", "dp0_x_attn"]], groups=data["market_id"])
    print("\n=== (3) FE + Cluster(market): dph ~ dp0 + dp0×log(1+attn_proxy) [FAST] ===")
    print(m3.summary())

    data["spike_sign"] = np.sign(data["dp0"])
    print("\nMean future change (dph) by spike sign:")
    print(data.groupby("spike_sign")["dph"].mean())

    out_csv = "polymarket_fast_panel.csv"
    data.to_csv(out_csv, index=False)
    print(f"\nSaved panel to: {out_csv}")


if __name__ == "__main__":
    main()