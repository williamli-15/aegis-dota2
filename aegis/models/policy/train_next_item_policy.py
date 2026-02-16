import argparse
import os
import json
import numpy as np
import polars as pl
import xgboost as xgb


def read_ids(path: str):
    with open(path, "r") as f:
        return [int(x.strip()) for x in f if x.strip()]


def load_split_df(path: str, match_ids):
    ids = pl.DataFrame({"match_id": match_ids})
    return (
        pl.scan_parquet(path)
        .join(ids.lazy(), on="match_id", how="inner")
        .collect()
    )


def topk_acc(proba: np.ndarray, y: np.ndarray, k: int) -> float:
    num_class = proba.shape[1]
    if num_class == 0:
        return float("nan")
    k = max(1, min(k, num_class))  # 防止 k<=0 或 k>num_class
    topk = np.argpartition(-proba, kth=k - 1, axis=1)[:, :k]
    hit = (topk == y[:, None]).any(axis=1)
    return float(hit.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/next_item_samples.parquet")
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--out_dir", default="artifacts/next_item_policy_xgb")
    ap.add_argument("--max_train_rows", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--use_gpu", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    # ----------------------------
    # Load splits
    # ----------------------------
    train_ids = read_ids(os.path.join(args.splits_dir, "train.txt"))
    val_ids = read_ids(os.path.join(args.splits_dir, "val.txt"))
    test_ids = read_ids(os.path.join(args.splits_dir, "test.txt"))

    df_tr = load_split_df(args.data, train_ids)
    df_va = load_split_df(args.data, val_ids)
    df_te = load_split_df(args.data, test_ids)

    # ----------------------------
    # Label space (use FULL train, before sampling)
    # ----------------------------
    labels = sorted(df_tr["y_next_item"].unique().to_list())
    label_to_id = {k: i for i, k in enumerate(labels)}
    num_class = len(labels)
    print(f"[info] full train rows={df_tr.height}, unique labels={num_class}")

    # ----------------------------
    # Filter val/test rows with unseen labels (w.r.t full train)
    # ----------------------------
    seen_labels = labels  # ✅ 用 list 更稳

    def filter_unseen(df: pl.DataFrame, split_name: str) -> pl.DataFrame:
        kept = df.filter(pl.col("y_next_item").is_in(seen_labels))
        dropped = df.height - kept.height
        if dropped > 0:
            denom = max(df.height, 1)
            print(
                f"[warn] {split_name}: dropped {dropped} rows with unseen labels "
                f"({dropped/denom:.2%} of split)"
            )
        return kept

    df_va = filter_unseen(df_va, "val")
    df_te = filter_unseen(df_te, "test")

    if df_va.height == 0:
        raise RuntimeError(
            "val split became empty after filtering unseen labels. "
            "This means val labels are not covered by train label space. "
            "Try: (1) check split files, (2) avoid too aggressive train sampling, "
            "(3) ensure train covers the intended patch/item vocabulary."
        )
    if df_te.height == 0:
        raise RuntimeError(
            "test split became empty after filtering unseen labels. "
            "This means test labels are not covered by train label space."
        )

    # ----------------------------
    # Sample train for speed (after label space is fixed)
    # ----------------------------
    if args.max_train_rows and df_tr.height > args.max_train_rows:
        df_tr = df_tr.sample(n=args.max_train_rows, seed=args.seed)
    print(f"[info] sampled train rows={df_tr.height}")

    # ----------------------------
    # Feature engineering
    # ----------------------------
    def featurize(df: pl.DataFrame):
        drop = {
            "match_id",
            "patch",
            "t",
            "first_t",
            "dt_to_buy",
            "y_next_item",
        }
        cols = [c for c in df.columns if c not in drop]
        X = df.select(cols).to_pandas()
        y = np.array(
            [label_to_id[v] for v in df["y_next_item"].to_list()],
            dtype=np.int32,
        )
        return cols, X, y

    feat_cols, X_tr, y_tr = featurize(df_tr)
    _, X_va, y_va = featurize(df_va)
    _, X_te, y_te = featurize(df_te)

    # ----------------------------
    # Model params (CPU/GPU unified)
    # ----------------------------
    params = dict(
        n_estimators=1200,          # upper bound; early stopping will cut earlier
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=num_class,
        random_state=args.seed,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=50,   # xgboost 3.x: put here (not in fit)
    )
    if args.use_gpu:
        params["device"] = "cuda"

    clf = xgb.XGBClassifier(**params)

    # ----------------------------
    # Train with GPU fallback
    # ----------------------------
    try:
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    except Exception as e:
        if args.use_gpu:
            print("[warn] GPU training failed, fallback to CPU.")
            print("       Error:", str(e)[:200])
            params.pop("device", None)
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        else:
            raise

    # ----------------------------
    # Print training info
    # ----------------------------
    best_iter = getattr(clf, "best_iteration", None)
    best_score = getattr(clf, "best_score", None)
    print("\n[info] Training complete")
    print(f"[info] best_iteration = {best_iter}")
    print(f"[info] best_score     = {best_score}")

    cfg = clf.get_booster().save_config()
    print("\n[info] Booster config (snippet):")
    print(cfg[:800], "...\n")

    # ----------------------------
    # Evaluation
    # ----------------------------
    proba = clf.predict_proba(X_te)

    acc1 = float((proba.argmax(axis=1) == y_te).mean())
    acc5 = topk_acc(proba, y_te, 5)
    acck = topk_acc(proba, y_te, args.topk)

    print("== Next-item policy (TEST) ==")
    print(
        f"Top1={acc1:.4f}  "
        f"Top5={acc5:.4f}  "
        f"Top{min(args.topk, proba.shape[1])}={acck:.4f}  "
        f"classes={num_class}"
    )

    # ----------------------------
    # Save artifacts
    # ----------------------------
    clf.save_model(os.path.join(args.out_dir, "next_item_xgb.json"))

    with open(os.path.join(args.out_dir, "label_to_id.json"), "w") as f:
        json.dump(label_to_id, f, indent=2)

    with open(os.path.join(args.out_dir, "id_to_label.json"), "w") as f:
        json.dump({str(i): k for k, i in label_to_id.items()}, f, indent=2)

    with open(os.path.join(args.out_dir, "feature_cols.json"), "w") as f:
        json.dump(feat_cols, f, indent=2)

    print("\nSaved to:", args.out_dir)


if __name__ == "__main__":
    main()
