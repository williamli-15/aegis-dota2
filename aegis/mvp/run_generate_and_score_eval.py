"""
魔法版本（generate-and-score eval）
python -m aegis.mvp.run_generate_and_score_eval --n_eval 5000 --topk 20

（不加 --use_gpu 就是 CPU）
"""

"""
日常动作打分（用 v2 feasible 版）

python -m aegis.mvp.score_actions_v2 \
  --match_id 8607168614 --t 900 --team 0 --mode both --topk 15 \
  --feasible --fetch_constants --min_item_cost 2000 --time_band_sec 900
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
    # p_radiant_win: shape (N,)
    return p_radiant_win if team == 0 else (1.0 - p_radiant_win)


def topk_acc_from_proba(proba: np.ndarray, y: np.ndarray, k: int) -> float:
    proba = np.asarray(proba)
    if proba.ndim != 2:
        raise ValueError(f"proba must be 2D (n,C), got {proba.shape}")
    C = proba.shape[1]
    if C == 0 or proba.shape[0] == 0:
        return float("nan")
    k = max(1, min(k, C))
    topk = np.argpartition(-proba, kth=k - 1, axis=1)[:, :k]
    hit = (topk == y[:, None]).any(axis=1)
    return float(hit.mean())


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

    # label_to_id for y_true mapping (policy classes)
    label_to_id = {v: k for k, v in id_to_label.items()}

    # Filter rows whose true label not in policy label space (prevents KeyError)
    # (你训练脚本里已经尽量避免了，这里再加一道保险)
    before = df.height
    df = df.filter(pl.col("y_next_item").is_in(list(label_to_id.keys())))
    dropped = before - df.height
    if dropped > 0:
        print(f"[warn] dropped {dropped} test rows with labels unseen by policy")
    if df.height == 0:
        raise RuntimeError("All test rows dropped because labels are unseen by policy")

    # ----------------------------
    # Load value model (win head)
    # ----------------------------
    with open(args.value_features, "r") as f:
        val_cols = [x.strip() for x in f if x.strip()]

    win_b = load_booster(os.path.join(args.value_dir, "win_xgb.json"))

    # Optional GPU for inference: XGBoost 3.x uses `device` on Booster via parameter
    # (policy/value models saved from training may already include device config; we set it explicitly)
    if args.use_gpu:
        try:
            policy.set_param({"device": "cuda"})
            win_b.set_param({"device": "cuda"})
            print("[info] inference device=cuda")
        except Exception as e:
            print("[warn] failed to set device=cuda, continue on CPU. Error:", str(e)[:200])
    else:
        print("[info] inference device=cpu")

    # ----------------------------
    # Build policy input (numpy) and predict proba
    # ----------------------------
    # fill nulls to 0; keep float32 for speed
    X_pol = (
        df.select([pl.col(c).fill_null(0).cast(pl.Float32) for c in pol_cols])
        .to_numpy()
    )
    # Faster than DMatrix for inference
    proba = policy.inplace_predict(X_pol)  # shape (n, C)
    
    proba = np.asarray(proba)

    C_expected = len(id_to_label)
    if proba.ndim == 1:
        # 兼容某些版本 multi:softprob 扁平输出 (n*C,)
        if proba.size % C_expected != 0:
            raise RuntimeError(f"policy proba shape unexpected: {proba.shape}, "
                            f"C_expected={C_expected}")
        proba = proba.reshape(-1, C_expected)
        

    # y_true class ids
    y_true = np.array([label_to_id[v] for v in df["y_next_item"].to_list()], dtype=np.int32)

    # topk candidates from policy
    C = proba.shape[1]
    k = max(1, min(args.topk, C))
    topk_idx = np.argpartition(-proba, kth=k - 1, axis=1)[:, :k]

    hitk = topk_acc_from_proba(proba, y_true, args.topk)

    # ----------------------------
    # Build base value features matrix (numpy) once
    # ----------------------------
    # Note: value features must match val_cols order exactly
    X_val_base = (
        df.select([pl.col(c).fill_null(0).cast(pl.Float32) for c in val_cols])
        .to_numpy()
        .astype(np.float32, copy=False)
    )

    teams = df["team"].to_numpy().astype(np.int32, copy=False)
    true_items = df["y_next_item"].to_list()

    # Precompute indices for fast feature toggling
    val_idx = {c: i for i, c in enumerate(val_cols)}

    # item -> (rad_idx, dire_idx, diff_idx); None if column not present
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
        # 如果 value 特征里根本没有这个 item 的列，就跳过（不会报错）
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

    # ----------------------------
    # Generate-and-score (FAST)
    # ----------------------------
    # Base win probs in one shot (n,)
    p0_all = np.asarray(win_b.inplace_predict(X_val_base)).reshape(-1)
    base_tw_all = np.where(teams == 0, p0_all, 1.0 - p0_all)

    chosen_hit = 0
    uplift_vs_true = []
    uplift_vs_base = []

    # For speed: precompute candidate items per row (strings)
    # (topk_idx has shape (n, k))
    n = df.height

    for i in range(n):
        team = int(teams[i])
        true_item = true_items[i]

        # candidate items from policy
        cand_ids = topk_idx[i]
        cand_items = [id_to_label[int(cid)] for cid in cand_ids]

        # always include true item
        if true_item not in cand_items:
            cand_items.append(true_item)

        K = len(cand_items)

        # Build candidate feature batch: repeat base row K times
        Xcand = np.repeat(X_val_base[i:i+1], K, axis=0)

        # Apply each candidate item in-place
        for j, it in enumerate(cand_items):
            apply_item_np(Xcand[j], it, team)

        # Predict radiant win prob for all candidates in ONE call
        p_cand = np.asarray(win_b.inplace_predict(Xcand)).reshape(-1)
        tw_cand = team_win_prob_vec(team, p_cand)

        best_j = int(np.argmax(tw_cand))
        best_tw = float(tw_cand[best_j])
        best_item = cand_items[best_j]

        # true item tw: find its index in cand_items (we ensured it exists)
        true_j = cand_items.index(true_item)
        true_tw = float(tw_cand[true_j])

        if best_item == true_item:
            chosen_hit += 1

        uplift_vs_true.append(best_tw - true_tw)
        uplift_vs_base.append(best_tw - float(base_tw_all[i]))

    print("\n== Generate-and-Score eval (TEST) ==")
    print(f"policy_hit@{k}: {hitk:.4f}  (requested topk={args.topk}, num_class={C})")
    print(f"g&s_hit@1: {chosen_hit/n:.4f}")
    print(f"avg_uplift_vs_true (proxy): {float(np.mean(uplift_vs_true)):+.4f}")
    print(f"avg_uplift_vs_base (proxy): {float(np.mean(uplift_vs_base)):+.4f}")


if __name__ == "__main__":
    main()
