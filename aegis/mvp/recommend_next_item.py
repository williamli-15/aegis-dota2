"""
python -m aegis.mvp.recommend_next_item \
  --match_id 8607168614 --t 900 --team 0 \
  --feasible --fetch_constants --min_item_cost 2000 --time_band_sec 900
"""

import argparse, os, json
import numpy as np
import polars as pl
import xgboost as xgb

def load_lines(p):
    with open(p, "r") as f:
        return [x.strip() for x in f if x.strip()]

def load_booster(p):
    b = xgb.Booster()
    b.load_model(p)
    return b

def load_row(value_parquet: str, match_id: int, t: int) -> dict:
    df = (pl.scan_parquet(value_parquet)
          .filter((pl.col("match_id")==match_id) & (pl.col("t")==t))
          .collect())
    if df.height == 0:
        ts = (pl.scan_parquet(value_parquet)
              .filter(pl.col("match_id")==match_id)
              .select(["t"]).collect())
        arr = ts["t"].to_numpy()
        t = int(arr[np.argmin(np.abs(arr - t))])
        df = (pl.scan_parquet(value_parquet)
              .filter((pl.col("match_id")==match_id) & (pl.col("t")==t))
              .collect())
        print(f"[warn] using closest t={t}")
    return df.row(0, named=True)

def team_win_prob(team: int, p_radiant: float) -> float:
    return p_radiant if team==0 else (1.0 - p_radiant)

def load_item_cost(constants_path: str, fetch: bool=False) -> dict[str,int]:
    def _cost(v):
        c = v.get("cost", 0)
        return int(c) if c is not None else 0
    if os.path.exists(constants_path):
        with open(constants_path, "r") as f:
            data = json.load(f)
        return {k:_cost(v) for k,v in data.items()}
    if not fetch:
        return {}
    import requests
    data = requests.get("https://api.opendota.com/api/constants/items", timeout=30).json()
    os.makedirs(os.path.dirname(constants_path), exist_ok=True)
    with open(constants_path, "w") as f:
        json.dump(data, f)
    return {k:_cost(v) for k,v in data.items()}

def compute_item_t50(events_glob: str, vocab: list[str], out_cache: str) -> dict[str,float]:
    if os.path.exists(out_cache):
        with open(out_cache, "r") as f:
            return {k: float(v) for k,v in json.load(f).items()}
    import duckdb
    con = duckdb.connect()
    in_list = ",".join("'" + v.replace("'", "''") + "'" for v in vocab)
    df = con.execute(f"""
    WITH first_buy AS (
      SELECT match_id, team, key, MIN(t) AS first_t
      FROM read_parquet('{events_glob}')
      WHERE event_type='purchase' AND t >= 0 AND key IN ({in_list})
      GROUP BY 1,2,3
    )
    SELECT key, quantile_cont(first_t, 0.5) AS t50_first_t
    FROM first_buy
    GROUP BY 1
    """).df()
    m = {r["key"]: float(r["t50_first_t"]) for _, r in df.iterrows()}
    os.makedirs(os.path.dirname(out_cache), exist_ok=True)
    with open(out_cache, "w") as f:
        json.dump(m, f)
    return m

def set_item(feat: np.ndarray, item: str, team: int, col_idx: dict) -> None:
    rad = col_idx.get(f"rad_has_{item}")
    dire = col_idx.get(f"dire_has_{item}")
    diff = col_idx.get(f"diff_has_{item}")
    if rad is None and dire is None and diff is None:
        return
    if team==0:
        if rad is not None: feat[rad] = 1.0
    else:
        if dire is not None: feat[dire] = 1.0
    if diff is not None:
        r = feat[rad] if rad is not None else 0.0
        d = feat[dire] if dire is not None else 0.0
        feat[diff] = r - d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match_id", type=int, required=True)
    ap.add_argument("--t", type=int, default=900)
    ap.add_argument("--team", type=int, default=0, choices=[0,1])
    ap.add_argument("--topk_policy", type=int, default=20)
    ap.add_argument("--topn_out", type=int, default=10)

    ap.add_argument("--value_parquet", default="data/processed/value_samples_v1_h180.parquet")
    ap.add_argument("--value_dir", default="artifacts/value_xgb_v1_h180")
    ap.add_argument("--value_features", default="artifacts/value_xgb_v1_h180/features.txt")

    ap.add_argument("--policy_dir", default="artifacts/next_item_policy_xgb")
    ap.add_argument("--events_glob", default="data/processed/events/*.parquet")
    ap.add_argument("--item_vocab", default="data/processed/item_vocab_v1.txt")
    ap.add_argument("--item_t50_cache", default="data/processed/item_t50_cache.json")
    ap.add_argument("--item_constants", default="data/processed/opendota_items_constants.json")

    ap.add_argument("--feasible", action="store_true")
    ap.add_argument("--fetch_constants", action="store_true")
    ap.add_argument("--min_item_cost", type=int, default=2000)
    ap.add_argument("--time_band_sec", type=int, default=900)
    ap.add_argument("--gold_buffer", type=float, default=800.0)
    args = ap.parse_args()

    # load feature row
    row = load_row(args.value_parquet, args.match_id, args.t)
    t_used = int(row["t"])

    # load policy
    policy = xgb.Booster()
    policy.load_model(os.path.join(args.policy_dir, "next_item_xgb.json"))
    with open(os.path.join(args.policy_dir, "feature_cols.json"), "r") as f:
        pol_cols = json.load(f)
    with open(os.path.join(args.policy_dir, "id_to_label.json"), "r") as f:
        id_to_label = {int(k): v for k,v in json.load(f).items()}

    # build policy input (polars row -> numpy)
    pol_feat = np.array([float(row.get(c, 0.0)) for c in pol_cols], dtype=np.float32)
    # inject team (policy expects it)
    if "team" in pol_cols:
        pol_feat[pol_cols.index("team")] = float(args.team)

    proba = np.asarray(policy.inplace_predict(pol_feat.reshape(1,-1)))
    Cexp = len(id_to_label)
    if proba.ndim == 1:
        proba = proba.reshape(-1, Cexp)
    proba = proba.reshape(-1)
    k = min(args.topk_policy, proba.shape[0])
    topk_idx = np.argpartition(-proba, kth=k-1)[:k]
    cand_items = [id_to_label[int(i)] for i in topk_idx]
    # de-dup preserve order
    seen=set(); cand_items=[x for x in cand_items if not (x in seen or seen.add(x))]

    # load value win model
    val_cols = load_lines(args.value_features)
    win_b = load_booster(os.path.join(args.value_dir, "win_xgb.json"))
    col_idx = {c:i for i,c in enumerate(val_cols)}
    base_vec = np.array([float(row.get(c,0.0)) for c in val_cols], dtype=np.float32)

    # base score
    p0 = float(np.asarray(win_b.inplace_predict(base_vec.reshape(1,-1))).reshape(-1)[0])
    base_tw = team_win_prob(args.team, p0)

    # feasibility metadata
    vocab = load_lines(args.item_vocab)
    item_t50 = compute_item_t50(args.events_glob, vocab, args.item_t50_cache) if args.time_band_sec>0 else {}
    item_cost = load_item_cost(args.item_constants, fetch=args.fetch_constants) if (args.feasible or args.min_item_cost>0) else {}

    gold_max = float(row.get("rad_gold_max",0.0) if args.team==0 else row.get("dire_gold_max",0.0))

    # filter + score
    scored=[]
    for it in cand_items:
        cost = item_cost.get(it, 0)
        if args.min_item_cost>0 and cost < args.min_item_cost:
            continue
        if args.time_band_sec>0:
            t50 = item_t50.get(it)
            if t50 is not None and abs(t_used - t50) > args.time_band_sec:
                continue
        if args.feasible and cost>0 and cost > (gold_max + args.gold_buffer):
            continue

        vec = base_vec.copy()
        set_item(vec, it, args.team, col_idx)
        p = float(np.asarray(win_b.inplace_predict(vec.reshape(1,-1))).reshape(-1)[0])
        tw = team_win_prob(args.team, p)
        scored.append((it, tw, tw-base_tw, cost))

    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)

    print(f"\n[Recommend] match={args.match_id} t={t_used} team={args.team}")
    print(f"base_team_win={base_tw:.4f}  gold_max={gold_max:.0f}")
    for i,(it,tw,dtw,cost) in enumerate(scored[:args.topn_out],1):
        print(f"{i:02d}. {it:20s}  win={tw:.4f}  Δwin={dtw:+.4f}  cost={cost}")

if __name__ == "__main__":
    main()
