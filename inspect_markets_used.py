import requests
import pandas as pd

GAMMA = "https://gamma-api.polymarket.com"

df = pd.read_csv("polymarket_research_panel.csv")
market_ids = df["market_id"].unique()

market_info = []

for mid in market_ids:
    r = requests.get(f"{GAMMA}/markets/{mid}")
    if r.status_code == 200:
        m = r.json()
        market_info.append({
            "market_id": mid,
            "question": m.get("question"),
            "liquidity": m.get("liquidity"),
            "volume": m.get("volume"),
            "endDate": m.get("endDateIso"),
            "category": m.get("category")
        })

info_df = pd.DataFrame(market_info)
info_df.to_csv("polymarket_markets_used.csv", index=False)

print(info_df.head())
print("\nTotal markets documented:", len(info_df))