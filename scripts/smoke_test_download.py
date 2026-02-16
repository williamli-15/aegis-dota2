import json
import time
import requests

API_KEY = os.environ["OPENDOTA_API_KEY"]
BASE = "https://api.opendota.com/api/matches/{}"
OUT_JSONL = "smoke_matches.jsonl"

# 你要 smoke test 的 2 个 match_id（第一个是你消息里给的示例）
MATCH_IDS = [
    8692677753,
    8692534229,
]

TIMEOUT_SEC = 30
SLEEP_SEC = 0.25


def is_parsed(match: dict) -> bool:
    od = match.get("od_data") or {}
    # OpenDota 明确标记
    if od.get("has_parsed") is True:
        return True
    # 兜底：parsed 一般会有这些字段（至少不应为 None）
    if match.get("teamfights") is not None and match.get("objectives") is not None:
        return True
    return False


def fetch_match(match_id: int) -> dict:
    url = BASE.format(match_id)
    r = requests.get(url, params={"api_key": API_KEY}, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    return r.json()


def main():
    ok = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for mid in MATCH_IDS:
            try:
                data = fetch_match(mid)
            except Exception as e:
                print(f"[FAIL] match_id={mid} err={e}")
                continue

            parsed = is_parsed(data)
            print(
                "[OK]" if parsed else "[NOT_PARSED]",
                f"match_id={data.get('match_id')} "
                f"patch={data.get('patch')} "
                f"version={data.get('version')} "
                f"duration={data.get('duration')} "
                f"has_parsed={(data.get('od_data') or {}).get('has_parsed')} "
                f"teamfights_type={type(data.get('teamfights')).__name__} "
                f"objectives_type={type(data.get('objectives')).__name__}"
            )

            if parsed:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()
                ok += 1

            time.sleep(SLEEP_SEC)

    print(f"\nSaved {ok} parsed matches to {OUT_JSONL}")


if __name__ == "__main__":
    main()
