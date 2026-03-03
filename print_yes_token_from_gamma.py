import requests, json

GAMMA = "https://gamma-api.polymarket.com"

# Pull a batch and find the first market that has clobTokenIds
batch = requests.get(f"{GAMMA}/markets", params={"limit": 200, "offset": 0}, timeout=30).json()

for m in batch:
    token_raw = m.get("clobTokenIds")
    if not token_raw:
        continue

    # must be orderbook-enabled to work with clob endpoints
    if not bool(m.get("enableOrderBook", False)):
        continue

    # parse token ids
    try:
        token_ids = json.loads(token_raw) if isinstance(token_raw, str) else token_raw
    except Exception:
        continue

    if not isinstance(token_ids, list) or len(token_ids) < 2:
        continue

    outcomes_raw = m.get("outcomes")
    outcomes = None
    if isinstance(outcomes_raw, str):
        try:
            outcomes = json.loads(outcomes_raw)
        except Exception:
            outcomes = None
    elif isinstance(outcomes_raw, list):
        outcomes = outcomes_raw

    yes_idx = 0
    if isinstance(outcomes, list):
        for i, o in enumerate(outcomes):
            if isinstance(o, str) and o.strip().lower() == "yes":
                yes_idx = i
                break

    yes_token = str(token_ids[yes_idx])

    print("Question:", m.get("question"))
    print("enableOrderBook:", m.get("enableOrderBook"))
    print("endDate:", m.get("endDateIso") or m.get("endDate"))
    print("YES token id:", yes_token)
    break
else:
    print("No CLOB-enabled market with clobTokenIds found in this batch. Try increasing offset.")