import csv
import json
import os
import time
from typing import Dict, Any, Iterable, Set, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import requests

API_KEY = os.environ["OPENDOTA_API_KEY"]

BASE = "https://api.opendota.com/api/matches/{}"

IN_CSV = "match_ids.csv"
OUT_JSONL = "matches_parsed_7.38_7.40_v22.jsonl"
OUT_BAD = "skipped_or_failed.txt"

TIMEOUT_SEC = 30

# 3000/min ~= 50/sec。实际吞吐取决于RTT，所以用并发“填满”。
MAX_WORKERS = 24          # 先用 16/24/32 试，别一上来 200
MAX_RPS = 45              # 留点余量，避免429（50是理论上限）
WRITE_FLUSH_EVERY = 200   # 每200条刷一次盘

# ---- 简单的全局限速器（跨线程） ----
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


def iter_match_ids(path: str) -> Iterable[int]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            v = row[0].strip()
            if v.lower() in ("match_id", "matchid"):
                continue
            yield int(v)


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


def is_parsed(match: Dict[str, Any]) -> bool:
    od = match.get("od_data") or {}
    if od.get("has_parsed") is True:
        return True
    # fallback：这两个大字段存在通常也意味着 parsed
    return (match.get("teamfights") is not None) and (match.get("objectives") is not None)


def fetch_one(session: requests.Session, match_id: int) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
    # 限速（全局）
    rate_limit()

    url = BASE.format(match_id)
    try:
        r = session.get(url, params={"api_key": API_KEY}, timeout=TIMEOUT_SEC)
    except Exception as e:
        return match_id, None, f"request_error:{e}"

    if r.status_code == 429:
        return match_id, None, "rate_limited"
    if r.status_code != 200:
        return match_id, None, f"http_{r.status_code}:{r.text[:120]}"

    try:
        return match_id, r.json(), None
    except Exception as e:
        return match_id, None, f"json_error:{e}"


def main():
    target_ids = list(dict.fromkeys(iter_match_ids(IN_CSV)))
    target_total = len(target_ids)

    done = load_done_ids(OUT_JSONL)
    target_set = set(target_ids)
    done_in_target = done.intersection(target_set)

    todo = [mid for mid in target_ids if mid not in done_in_target]

    print(f"Target: {target_total} | Already saved: {len(done_in_target)} | To fetch: {len(todo)}")
    print(f"Workers={MAX_WORKERS}, MAX_RPS={MAX_RPS}")

    session = requests.Session()
    # 复用连接 + 适当扩大连接池
    adapter = requests.adapters.HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    session.mount("https://", adapter)

    saved = len(done_in_target)
    buf_out = []
    buf_bad = []

    # 失败重试策略：rate_limited / 短暂网络错误，最多重试3次
    def fetch_with_retry(mid: int):
        backoff = 0.5
        for attempt in range(4):
            _mid, data, err = fetch_one(session, mid)
            if data is not None:
                return _mid, data, None
            if err == "rate_limited":
                time.sleep(backoff)
                backoff *= 2
                continue
            if err and err.startswith("request_error"):
                time.sleep(backoff)
                backoff *= 2
                continue
            return _mid, None, err
        return mid, None, "rate_limited_retries_exhausted"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex, \
         open(OUT_JSONL, "a", encoding="utf-8") as out, \
         open(OUT_BAD, "a", encoding="utf-8") as bad:

        futures = {ex.submit(fetch_with_retry, mid): mid for mid in todo}

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
                # 严格过滤：parsed + version=22
                if not is_parsed(data):
                    buf_bad.append(f"{mid}\tSKIP\tnot_parsed\n")
                elif data.get("version") != 22:
                    buf_bad.append(f"{mid}\tSKIP\tversion={data.get('version')}\n")
                else:
                    buf_out.append(json.dumps(data, ensure_ascii=False) + "\n")
                    saved += 1

            # 批量写入
            if (len(buf_out) + len(buf_bad)) >= WRITE_FLUSH_EVERY:
                if buf_out:
                    out.writelines(buf_out)
                    buf_out.clear()
                if buf_bad:
                    bad.writelines(buf_bad)
                    buf_bad.clear()
                out.flush()
                bad.flush()
                print(f"Saved so far: {saved}/{target_total} | remaining futures: {len(futures)}")

        # flush remaining
        if buf_out:
            out.writelines(buf_out)
        if buf_bad:
            bad.writelines(buf_bad)
        out.flush()
        bad.flush()

    print(f"Done. Saved {saved}/{target_total} into {OUT_JSONL}.")


if __name__ == "__main__":
    main()
