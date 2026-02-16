"""
python -m aegis.mvp.run_generate_and_score_eval_v2 \
  --n_eval 5000 --topk 20 \
  --feasible --fetch_constants \
  --min_item_cost 2000 \
  --time_band_sec 900 \
  --gold_buffer 800
"""

"""
TODO:
cost adjust
"""


import argparse, os, json
import numpy as np
import polars as pl
import xgboost as xgb

def read_ids(path: str):
    with open(path, "r") as f:
        return [int(x.strip()) for x in f if x.strip()]

def load_booster(path: str) -> xgb.Booster:
    b = xgb.Booster()
    b.load_model(path)
    return b

def team_win_prob_vec(team: int, p_radiant_win: np.ndarray) -> np.ndarray:
    return p_radiant_win if team == 0 else (1.0 - p_radiant_win)

def topk_acc_from_proba(proba: np.ndarray, y: np.ndarray, k: int) -> float:
    proba = np.asarray(proba)
    C = proba.shape[1]
    k = max(1, min(k, C))
    topk = np.argpartition(-proba, kth=k - 1, axis=1)[:, :k]
    return float((topk == y[:, None]).any(axis=1).mean())

def load_item_cost(constants_path: str, fetch: bool=False) -> dict[str,int]:
    def _cost(v):
        c = v.get("cost", 0)
        return int(c) if c is not None else 0

    if os.path.exists(constants_path):
        with open(constants_path, "r") as f:
            data = json.load(f)
        return {k: _cost(v) for k, v in data.items()}

    if not fetch:
        return {}

    import requests
    url = "https://api.opendota.com/api/constants/items"
    data = requests.get(url, timeout=30).json()
    os.makedirs(os.path.dirname(constants_path), exist_ok=True)
    with open(constants_path, "w") as f:
        json.dump(data, f)
    return {k: _cost(v) for k, v in data.items()}

def compute_item_t50(events_glob: str, vocab: list[str], out_cache: str) -> dict[str,float]:
    if os.path.exists(out_cache):
        with open(out_cache, "r") as f:
            return {k: float(v) for k, v in json.load(f).items()}

    import duckdb
    import pandas as pd
    con = duckdb.connect()

    con.register("vocab_tbl", pd.DataFrame({"key": vocab}))

    df = con.execute(f"""
    WITH first_buy AS (
      SELECT e.match_id, e.team, e.key, MIN(e.t) AS first_t
      FROM read_parquet('{events_glob}') e
      JOIN vocab_tbl v ON e.key = v.key
      WHERE e.event_type='purchase' AND e.t >= 0
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/processed/next_item_samples.parquet")
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--policy_dir", default="artifacts/next_item_policy_xgb")
    ap.add_argument("--value_dir", default="artifacts/value_xgb_v1_h180")
    ap.add_argument("--value_features", default="artifacts/value_xgb_v1_h180/features.txt")
    ap.add_argument("--n_eval", type=int, default=5000)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_gpu", action="store_true")

    # --- new: feasible/time-band ---
    ap.add_argument("--feasible", action="store_true")
    ap.add_argument("--gold_buffer", type=float, default=800.0, help="allow cost <= gold_max + buffer")
    ap.add_argument("--min_item_cost", type=int, default=0)
    ap.add_argument("--time_band_sec", type=int, default=900)
    ap.add_argument("--fetch_constants", action="store_true")
    ap.add_argument("--item_constants", default="data/processed/opendota_items_constants.json")
    ap.add_argument("--events_glob", default="data/processed/events/*.parquet")
    ap.add_argument("--item_t50_cache", default="data/processed/item_t50_cache.json")
    ap.add_argument("--item_vocab", default="data/processed/item_vocab_v1.txt")

    args = ap.parse_args()
    np.random.seed(args.seed)

    # ----------------------------
    # Load test rows
    # ----------------------------
    test_ids = read_ids(os.path.join(args.splits_dir, "test.txt"))
    ids = pl.DataFrame({"match_id": test_ids})

    df = (
        pl.scan_parquet(args.samples)
        .join(ids.lazy(), on="match_id", how="inner")
        .collect()
    )
    if df.height == 0:
        raise RuntimeError("No test rows loaded")
    if df.height > args.n_eval:
        df = df.sample(n=args.n_eval, seed=args.seed)

    # ----------------------------
    # Load policy model + metadata
    # ----------------------------
    policy = xgb.Booster()
    policy.load_model(os.path.join(args.policy_dir, "next_item_xgb.json"))

    with open(os.path.join(args.policy_dir, "id_to_label.json"), "r") as f:
        id_to_label = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(args.policy_dir, "feature_cols.json"), "r") as f:
        pol_cols = json.load(f)

    label_to_id = {v: k for k, v in id_to_label.items()}

    before = df.height
    df = df.filter(pl.col("y_next_item").is_in(list(label_to_id.keys())))
    dropped = before - df.height
    if dropped > 0:
        print(f"[warn] dropped {dropped} test rows with labels unseen by policy")
    if df.height == 0:
        raise RuntimeError("All test rows dropped")

    # ----------------------------
    # Load value model (win head)
    # ----------------------------
    with open(args.value_features, "r") as f:
        val_cols = [x.strip() for x in f if x.strip()]
    win_b = load_booster(os.path.join(args.value_dir, "win_xgb.json"))

    if args.use_gpu:
        try:
            policy.set_param({"device": "cuda"})
            win_b.set_param({"device": "cuda"})
            print("[info] inference device=cuda")
        except Exception as e:
            print("[warn] failed to set device=cuda, continue on CPU:", str(e)[:200])
    else:
        print("[info] inference device=cpu")

    # ----------------------------
    # Load cost + t50 (for filtering)
    # ----------------------------
    item_cost = {}
    if args.feasible or args.min_item_cost > 0:
        item_cost = load_item_cost(args.item_constants, fetch=args.fetch_constants)

    item_t50 = {}
    if args.time_band_sec > 0:
        vocab = [x.strip() for x in open(args.item_vocab) if x.strip()]
        item_t50 = compute_item_t50(args.events_glob, vocab, args.item_t50_cache)

    # ----------------------------
    # Policy proba
    # ----------------------------
    X_pol = (
        df.select([pl.col(c).fill_null(0).cast(pl.Float32) for c in pol_cols])
        .to_numpy()
    )
    proba = np.asarray(policy.inplace_predict(X_pol))
    C_expected = len(id_to_label)
    if proba.ndim == 1:
        proba = proba.reshape(-1, C_expected)

    y_true = np.array([label_to_id[v] for v in df["y_next_item"].to_list()], dtype=np.int32)
    C = proba.shape[1]
    k = max(1, min(args.topk, C))
    topk_idx = np.argpartition(-proba, kth=k - 1, axis=1)[:, :k]
    hitk = topk_acc_from_proba(proba, y_true, args.topk)

    # ----------------------------
    # Base value matrix
    # ----------------------------
    X_val_base = (
        df.select([pl.col(c).fill_null(0).cast(pl.Float32) for c in val_cols])
        .to_numpy()
        .astype(np.float32, copy=False)
    )

    teams = df["team"].to_numpy().astype(np.int32, copy=False)
    t_arr = df["t"].to_numpy().astype(np.int32, copy=False)

    # use gold_max as feasibility proxy
    rad_gold_max = df["rad_gold_max"].to_numpy().astype(np.float32, copy=False)
    dire_gold_max = df["dire_gold_max"].to_numpy().astype(np.float32, copy=False)
    gold_max_team = np.where(teams == 0, rad_gold_max, dire_gold_max)

    true_items = df["y_next_item"].to_list()

    val_idx = {c: i for i, c in enumerate(val_cols)}
    item_to_cols = {}

    def get_item_col_idxs(item: str):
        if item in item_to_cols:
            return item_to_cols[item]
        rad = val_idx.get(f"rad_has_{item}")
        dire = val_idx.get(f"dire_has_{item}")
        diff = val_idx.get(f"diff_has_{item}")
        item_to_cols[item] = (rad, dire, diff)
        return rad, dire, diff

    def apply_item_np(xrow: np.ndarray, item: str, team: int):
        rad, dire, diff = get_item_col_idxs(item)
        if rad is None and dire is None and diff is None:
            return
        if team == 0:
            if rad is not None:
                xrow[rad] = 1.0
        else:
            if dire is not None:
                xrow[dire] = 1.0
        if diff is not None:
            r = xrow[rad] if rad is not None else 0.0
            d = xrow[dire] if dire is not None else 0.0
            xrow[diff] = r - d

    # Base win prob
    p0_all = np.asarray(win_b.inplace_predict(X_val_base)).reshape(-1)
    base_tw_all = np.where(teams == 0, p0_all, 1.0 - p0_all)

    # ----------------------------
    # Generate-and-score with filters
    # ----------------------------
    chosen_hit = 0
    n_valid_true = 0
    uplift_vs_true = []
    uplift_vs_base = []

    for i in range(df.height):
        team = int(teams[i])
        true_item = true_items[i]
        t_used = int(t_arr[i])
        gmax = float(gold_max_team[i])

        cand_ids = topk_idx[i]
        cand_items = [id_to_label[int(cid)] for cid in cand_ids]

        # de-dup preserve order
        seen = set()
        cand_items = [x for x in cand_items if not (x in seen or seen.add(x))]

        # apply filters
        filtered = []
        for it in cand_items:
            cost = item_cost.get(it, 0)
            if args.min_item_cost > 0 and cost < args.min_item_cost:
                continue
            if args.time_band_sec > 0:
                t50 = item_t50.get(it)
                if t50 is not None and abs(t_used - t50) > args.time_band_sec:
                    continue
            if args.feasible and cost > 0 and (cost > (gmax + args.gold_buffer)):
                continue
            filtered.append(it)

        # if nothing left, fallback to base
        if len(filtered) == 0:
            uplift_vs_base.append(0.0)
            continue

        # score candidates (batch)
        K = len(filtered)
        Xcand = np.repeat(X_val_base[i:i+1], K, axis=0)
        for j, it in enumerate(filtered):
            apply_item_np(Xcand[j], it, team)

        p_cand = np.asarray(win_b.inplace_predict(Xcand)).reshape(-1)
        tw_cand = team_win_prob_vec(team, p_cand)

        best_j = int(np.argmax(tw_cand))
        best_item = filtered[best_j]
        best_tw = float(tw_cand[best_j])

        uplift_vs_base.append(best_tw - float(base_tw_all[i]))

        # Only compute "vs_true" + hit@1 if true_item survived filters
        if true_item in filtered:
            n_valid_true += 1
            true_j = filtered.index(true_item)
            true_tw = float(tw_cand[true_j])
            uplift_vs_true.append(best_tw - true_tw)
            if best_item == true_item:
                chosen_hit += 1

    print("\n== Generate-and-Score eval (TEST, feasible/time-band) ==")
    print(f"policy_hit@{k}: {hitk:.4f}  (requested topk={args.topk}, num_class={C})")
    if n_valid_true > 0:
        print(f"g&s_hit@1 (only where true feasible): {chosen_hit/n_valid_true:.4f}  (n={n_valid_true})")
        print(f"avg_uplift_vs_true (proxy, feasible-only): {float(np.mean(uplift_vs_true)):+.4f}")
    else:
        print("g&s_hit@1: n_valid_true=0 (true item always filtered out)")
        print("avg_uplift_vs_true: n/a")
    print(f"avg_uplift_vs_base (proxy): {float(np.mean(uplift_vs_base)):+.4f}")

if __name__ == "__main__":
    main()
