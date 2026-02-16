import os
import json
import time
import threading
from typing import Dict, Any, Optional, Tuple, Set, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API_KEY = os.environ["OPENDOTA_API_KEY"]
BASE = "https://api.opendota.com/api/matches/{}"

OUT_JSONL = "matches_parsed_7.38_7.40_v22.jsonl"
OUT_BAD = "skipped_or_failed.txt"
OUT_RETRY_BAD = "retry_failed_still_bad.txt"   # 重试后仍失败的
TIMEOUT_SEC = 30

MAX_WORKERS = 24
MAX_RPS = 35                 # 重试阶段建议更保守一点
WRITE_FLUSH_EVERY = 100

_rate_lock = threading.Lock()
_next_allowed = 0.0

def rate_limit():
    global _next_allowed
    interval = 1.0 / MAX_RPS
    with _rate_lock:
        now = time.monotonic()
        if now < _next_allowed:
            time.sleep(_next_allowed - now)
        _next_allowed = max(_next_allowed, time.monotonic()) + interval

def load_done_ids(jsonl_path: str) -> Set[int]:
    done: Set[int] = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                mid = obj.get("match_id")
                if mid is not None:
                    done.add(int(mid))
            except Exception:
                continue
    return done

def parse_failed_ids(bad_path: str) -> List[int]:
    """
    只挑 FAIL 行：<match_id>\tFAIL\t<reason>
    """
    ids: List[int] = []
    with open(bad_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            if parts[1] != "FAIL":
                continue
            try:
                ids.append(int(parts[0]))
            except Exception:
                continue
    # 去重但保持顺序
    return list(dict.fromkeys(ids))

def is_parsed(match: Dict[str, Any]) -> bool:
    od = match.get("od_data") or {}
    if od.get("has_parsed") is True:
        return True
    return (match.get("teamfights") is not None) and (match.get("objectives") is not None)

def fetch_one(session: requests.Session, match_id: int) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
    rate_limit()
    url = BASE.format(match_id)
    try:
        r = session.get(url, params={"api_key": API_KEY}, timeout=TIMEOUT_SEC)
    except Exception as e:
        return match_id, None, f"request_error:{e}"

    if r.status_code == 429:
        return match_id, None, "rate_limited"
    if r.status_code >= 500:
        return match_id, None, f"http_{r.status_code}"
    if r.status_code != 200:
        return match_id, None, f"http_{r.status_code}:{r.text[:120]}"

    try:
        return match_id, r.json(), None
    except Exception as e:
        return match_id, None, f"json_error:{e}"

def fetch_with_retry(session: requests.Session, mid: int) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
    backoff = 0.5
    for attempt in range(6):  # 重试多一点
        _mid, data, err = fetch_one(session, mid)
        if data is not None:
            return _mid, data, None
        # 429 / 5xx / 网络抖动：都退避重试
        if err in ("rate_limited",) or (err and err.startswith("http_5")) or (err and err.startswith("request_error")):
            time.sleep(backoff)
            backoff = min(backoff * 2, 20)
            continue
        return _mid, None, err
    return mid, None, "retries_exhausted"

def main():
    done = load_done_ids(OUT_JSONL)
    failed_ids = parse_failed_ids(OUT_BAD)
    todo = [mid for mid in failed_ids if mid not in done]

    print(f"Failed IDs in {OUT_BAD}: {len(failed_ids)}")
    print(f"Already saved in {OUT_JSONL}: {len(done)}")
    print(f"To retry now: {len(todo)}")
    if not todo:
        return

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    session.mount("https://", adapter)

    buf_out, buf_bad = [], []
    saved_now = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex, \
         open(OUT_JSONL, "a", encoding="utf-8") as out, \
         open(OUT_RETRY_BAD, "a", encoding="utf-8") as bad:

        futures = {ex.submit(fetch_with_retry, session, mid): mid for mid in todo}
        for fut in as_completed(futures):
            mid = futures[fut]
            try:
                _mid, data, err = fut.result()
            except Exception as e:
                buf_bad.append(f"{mid}\tFAIL\tworker_exception:{e}\n")
                continue

            if data is None:
                buf_bad.append(f"{mid}\tFAIL\t{err}\n")
            else:
                if not is_parsed(data):
                    buf_bad.append(f"{mid}\tSKIP\tnot_parsed\n")
                elif data.get("version") != 22:
                    buf_bad.append(f"{mid}\tSKIP\tversion={data.get('version')}\n")
                else:
                    buf_out.append(json.dumps(data, ensure_ascii=False) + "\n")
                    saved_now += 1

            if (len(buf_out) + len(buf_bad)) >= WRITE_FLUSH_EVERY:
                if buf_out:
                    out.writelines(buf_out)
                    buf_out.clear()
                    out.flush()
                if buf_bad:
                    bad.writelines(buf_bad)
                    buf_bad.clear()
                    bad.flush()

    # flush tail
    if buf_out:
        with open(OUT_JSONL, "a", encoding="utf-8") as out:
            out.writelines(buf_out)
    if buf_bad:
        with open(OUT_RETRY_BAD, "a", encoding="utf-8") as bad:
            bad.writelines(buf_bad)

    print(f"Retry done. Newly saved: {saved_now}")

if __name__ == "__main__":
    main()
