import json
import time
import requests
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

GAMMA = "https://gamma-api.polymarket.com"
ORDERBOOK_SUBGRAPH = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"

# -----------------------------
# Config
# -----------------------------
KEYWORDS = [
    "taiwan", "china", "gaza", "israel", "ceasefire", "ukraine", "russia",
    "iran", "sanctions", "nato", "election", "war"
]

GAMMA_LIMIT = 200
GAMMA_MAX_PAGES = 100  # scans up to 20k markets

MAX_MARKETS_TOTAL = 150
MIN_VOLUME_USD = 0
MIN_LIQUIDITY = 0

RESAMPLE_FREQ = "1h"
K_STEPS = 3
H_STEPS = 12
MIN_POINTS_AFTER_RESAMPLE = 150

EVENTS_PAGE = 1000
MAX_EVENTS_PER_TOKEN = 40000
REQUEST_SLEEP = 0.12


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

def try_parse_json_list(x):
    if x is None:
        return None
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None

def gamma_list_markets(limit=200, offset=0):
    # Active markets only helps ensure CLOB/subgraph linkage
    r = requests.get(
        f"{GAMMA}/markets",
        params={"limit": limit, "offset": offset},
        timeout=30
    )
    r.raise_for_status()
    return r.json()

def extract_yes_token(market_obj):
    token_ids = try_parse_json_list(market_obj.get("clobTokenIds"))
    outcomes = try_parse_json_list(market_obj.get("outcomes"))

    if not token_ids or not isinstance(token_ids, list) or len(token_ids) < 2:
        return None

    yes_idx = 0
    if isinstance(outcomes, list):
        for i, o in enumerate(outcomes):
            if isinstance(o, str) and o.strip().lower() == "yes":
                yes_idx = i
                break

    return str(token_ids[yes_idx])

def gql(query, variables=None):
    payload = {"query": query, "variables": variables or {}}
    r = requests.post(ORDERBOOK_SUBGRAPH, json=payload, timeout=60)
    r.raise_for_status()
    out = r.json()
    if "errors" in out:
        raise RuntimeError(out["errors"])
    return out["data"]


# -----------------------------
# Subgraph pulls
# -----------------------------
def fetch_fills_for_token(token_id: str, max_events=20000):
    """
    Pull all OrderFilledEvents where the token appears either as makerAssetId or takerAssetId.
    We'll compute implied YES price from USDC/YES ratios.
    """
    all_rows = []
    skip = 0

    query = """
    query($first:Int!, $skip:Int!, $token:String!) {
      orderFilledEvents(
        first: $first,
        skip: $skip,
        orderBy: timestamp,
        orderDirection: asc,
        where: {
          or: [
            { makerAssetId: $token },
            { takerAssetId: $token }
          ]
        }
      ) {
        id
        timestamp
        maker
        taker
        makerAssetId
        takerAssetId
        makerAmountFilled
        takerAmountFilled
        fee
      }
    }
    """

    while True:
        data = gql(query, {"first": EVENTS_PAGE, "skip": skip, "token": token_id})
        rows = data.get("orderFilledEvents") or []
        if not rows:
            break

        all_rows.extend(rows)
        skip += EVENTS_PAGE

        if len(all_rows) >= max_events or len(rows) < EVENTS_PAGE:
            break

        time.sleep(REQUEST_SLEEP)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    # Convert types
    df["t"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="s", utc=True)
    df["makerAmountFilled"] = pd.to_numeric(df["makerAmountFilled"], errors="coerce")
    df["takerAmountFilled"] = pd.to_numeric(df["takerAmountFilled"], errors="coerce")
    df = df.dropna(subset=["t", "makerAmountFilled", "takerAmountFilled"])

    if df.empty:
        return None

    # Compute implied YES price:
    # If YES token is makerAssetId, maker amount is YES, taker amount is the other asset (likely USDC)
    # If YES token is takerAssetId, taker amount is YES, maker amount is the other asset (likely USDC)
    yes_is_maker = df["makerAssetId"].astype(str) == str(token_id)
    yes_is_taker = df["takerAssetId"].astype(str) == str(token_id)

    # price = other_amount / yes_amount
    price = np.where(
        yes_is_maker,
        df["takerAmountFilled"] / df["makerAmountFilled"],
        np.where(
            yes_is_taker,
            df["makerAmountFilled"] / df["takerAmountFilled"],
            np.nan
        )
    )

    df["p_raw"] = price
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["p_raw"])

    # In many Polymarket fills, this ratio should already be close to [0,1]
    # Clamp to [0,1] to avoid tiny decimal-scale mismatches breaking the model.
    df["p"] = df["p_raw"].clip(lower=0.0, upper=1.0)

    # Flow / attention measures
    df["amt_other"] = np.where(
        yes_is_maker, df["takerAmountFilled"],
        np.where(yes_is_taker, df["makerAmountFilled"], np.nan)
    )
    df["amt_other"] = pd.to_numeric(df["amt_other"], errors="coerce")

    df = df.dropna(subset=["p"]).sort_values("t")
    return df[["t", "p", "amt_other"]].reset_index(drop=True)


def resample_last_price_and_flow(fills_df: pd.DataFrame, freq="1h"):
    df = fills_df.set_index("t").sort_index()
    p = df["p"].resample(freq).last().ffill()
    n = df["p"].resample(freq).size().fillna(0)
    amt = df["amt_other"].resample(freq).sum().fillna(0)

    out = pd.DataFrame({"p": p, "n_fills": n, "amt": amt}).reset_index()
    out["t"] = pd.to_datetime(out["t"], utc=True)
    return out


def build_panel(prob_df: pd.DataFrame, k_steps: int, h_steps: int):
    p = prob_df["p"].to_numpy()
    t = prob_df["t"].reset_index(drop=True)
    n = prob_df["n_fills"].to_numpy()
    amt = prob_df["amt"].to_numpy()

    if len(p) <= (k_steps + h_steps + 5):
        return None

    idx = np.arange(k_steps, len(p) - h_steps)
    dp0 = p[idx] - p[idx - k_steps]
    dph = p[idx + h_steps] - p[idx]

    attn_fills = np.array([n[i - k_steps + 1: i + 1].sum() for i in idx])
    attn_amt = np.array([amt[i - k_steps + 1: i + 1].sum() for i in idx])

    return pd.DataFrame({
        "t": t.iloc[idx].to_numpy(),
        "dp0": dp0,
        "dph": dph,
        "attn_fills": attn_fills,
        "attn_amt": attn_amt,
    })


# -----------------------------
# Econometrics
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
# Main
# -----------------------------
def main():
    kw = [k.lower() for k in KEYWORDS]

    # Pull markets
    all_markets = []
    for page in range(GAMMA_MAX_PAGES):
        offset = page * GAMMA_LIMIT
        batch = gamma_list_markets(limit=GAMMA_LIMIT, offset=offset)
        if not isinstance(batch, list) or len(batch) == 0:
            break
        all_markets.extend(batch)
        time.sleep(REQUEST_SLEEP)

    print("Markets scanned from Gamma:", len(all_markets))

    selected = []
    for m in all_markets:
        q = (m.get("question") or "").lower()
        if not q:
            continue
        if not any(k in q for k in kw):
            continue
        if not bool(m.get("enableOrderBook", False)):
            continue

        vol = safe_float(m.get("volumeNum"), safe_float(m.get("volume"), 0.0))
        liq = safe_float(m.get("liquidityNum"), safe_float(m.get("liquidity"), 0.0))
        if vol < MIN_VOLUME_USD or liq < MIN_LIQUIDITY:
            continue

        yes_token = extract_yes_token(m)
        if yes_token is None:
            continue

        selected.append({
            "market_id": str(m.get("id")),
            "question": m.get("question", ""),
            "volume": vol,
            "liquidity": liq,
            "yes_token": yes_token,
        })

    mkts = pd.DataFrame(selected).drop_duplicates("market_id")
    mkts = mkts.sort_values(["liquidity", "volume"], ascending=False).head(MAX_MARKETS_TOTAL).reset_index(drop=True)

    print("Keyword-matched, orderbook-enabled markets:", len(mkts))
    if len(mkts) == 0:
        print("No markets matched. Broaden KEYWORDS.")
        return

    panels = []
    kept = 0

    for _, m in tqdm(mkts.iterrows(), total=len(mkts)):
        token = m["yes_token"]

        fills = fetch_fills_for_token(token, max_events=MAX_EVENTS_PER_TOKEN)
        if fills is None or len(fills) < 200:
            continue

        rs = resample_last_price_and_flow(fills, freq=RESAMPLE_FREQ)
        if len(rs) < MIN_POINTS_AFTER_RESAMPLE:
            continue

        panel = build_panel(rs, k_steps=K_STEPS, h_steps=H_STEPS)
        if panel is None or len(panel) < 200:
            continue

        panel["market_id"] = m["market_id"]
        panel["liquidity"] = float(m["liquidity"])
        panels.append(panel)
        kept += 1

        time.sleep(REQUEST_SLEEP)

    if not panels:
        print("No usable markets after pulling fill events (even with corrected schema).")
        print("If this happens, likely the computed p is not in [0,1] due to decimals; we’ll add decimals correction.")
        return

    data = pd.concat(panels, ignore_index=True)
    print("\nUsable markets:", kept)
    print("Total obs:", len(data))

    # (1) Baseline
    fe1 = demean_by_group(data, "market_id", ["dph", "dp0"])
    m1 = ols_cluster(fe1["dph"], fe1[["dp0"]], groups=data["market_id"])
    print("\n=== (1) FE + Cluster(market): dph ~ dp0 ===")
    print(m1.summary())

    # (2) Liquidity mechanism
    data["log_liq"] = np.log(np.clip(data["liquidity"], 1e-9, None))
    data["dp0_x_logliq"] = data["dp0"] * data["log_liq"]
    fe2 = demean_by_group(data, "market_id", ["dph", "dp0", "dp0_x_logliq"])
    m2 = ols_cluster(fe2["dph"], fe2[["dp0", "dp0_x_logliq"]], groups=data["market_id"])
    print("\n=== (2) FE + Cluster(market): dph ~ dp0 + dp0×log(liquidity) ===")
    print(m2.summary())

    # (3) Attention mechanism (fills)
    data["log_attn"] = np.log1p(data["attn_fills"])
    data["dp0_x_attn"] = data["dp0"] * data["log_attn"]
    fe3 = demean_by_group(data, "market_id", ["dph", "dp0", "dp0_x_attn"])
    m3 = ols_cluster(fe3["dph"], fe3[["dp0", "dp0_x_attn"]], groups=data["market_id"])
    print("\n=== (3) FE + Cluster(market): dph ~ dp0 + dp0×log(1+fills) ===")
    print(m3.summary())

    out_csv = "polymarket_research_panel.csv"
    data.to_csv(out_csv, index=False)
    print(f"\nSaved panel to: {out_csv}")


if __name__ == "__main__":
    main()