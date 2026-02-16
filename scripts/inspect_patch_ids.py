import os
import requests

API_KEY = os.environ["OPENDOTA_API_KEY"]

# !!! 把这里的 0 换成你 SQL DISTINCT ON 查出来的 match_id
match_ids = {
    "7.38": 8303667672,
    "7.39": 8607145322,
    "7.40": 8692695136,
}

for pv, mid in match_ids.items():
    url = f"https://api.opendota.com/api/matches/{mid}"
    r = requests.get(url, params={"api_key": API_KEY}, timeout=30)

    print("\n===", pv, "match_id=", mid, "status=", r.status_code, "===")

    # 先检查是否请求成功
    if r.status_code != 200:
        print("HTTP ERROR:", r.text[:300])
        continue

    j = r.json()

    # 有些错误会是 200 但带 error 字段（保险）
    if isinstance(j, dict) and "error" in j:
        print("API ERROR:", j["error"])
        continue

    print(
        "patch_id:", j.get("patch"),
        "| version:", j.get("version"),
        "| start_time:", j.get("start_time"),
        "| duration:", j.get("duration"),
    )
