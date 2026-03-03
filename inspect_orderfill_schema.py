import requests

ORDERBOOK_SUBGRAPH = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"

def gql(query, variables=None):
    r = requests.post(ORDERBOOK_SUBGRAPH, json={"query": query, "variables": variables or {}}, timeout=60)
    r.raise_for_status()
    out = r.json()
    if "errors" in out:
        raise RuntimeError(out["errors"])
    return out["data"]

q = """
query {
  __type(name: "OrderFilledEvent") {
    name
    fields {
      name
      type { name kind ofType { name kind } }
    }
  }
}
"""

data = gql(q)
t = data["__type"]
print("Type:", t["name"])
print("Fields:")
for f in t["fields"]:
    typ = f["type"]
    # unwrap type name
    tn = typ.get("name") or (typ.get("ofType") or {}).get("name")
    tk = typ.get("kind") or (typ.get("ofType") or {}).get("kind")
    print("-", f["name"], ":", tk, tn)