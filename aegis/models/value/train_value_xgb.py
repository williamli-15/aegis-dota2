
"""
# v0 / v1 到底是什么（一句话版）

## Value v0（baseline，最原始）

**只用“经济/经验/补刀/当前优势曲线”的聚合特征**来预测：

* `y_win`
* `y_dgold_adv@3m`
* `y_dxp_adv@3m`

对应脚本/产物：

* 由 `aegis/etl/make_value_samples.py` 生成 → `value_samples_h180.parquet`
* 用 `train_value_xgb.py` 训练 → `artifacts/value_xgb_h180/`

> v0 = “只看经济面板 + advantage 曲线”的价值模型（不看你有没有 BKB、有没有视野、最近打没打架）

---

## Value v1（你现在用的）

在 v0 的基础上 **额外拼进三块信息**：

* team items（`rad_has_* / dire_has_* / diff_has_*`）
* wards（active obs/sen + 1m/5m 放/拆）
* skirmish（kills/deaths + fights counts + time_since_fight）

对应脚本/产物：

* `aegis/etl/make_value_samples_v1.py` 生成 → `value_samples_v1_h180.parquet`
* 训练好的模型就是你现在的 → `artifacts/value_xgb_v1_h180/`

> v1 = “经济 + 物品 + 视野 + 近期冲突节奏”的价值模型（更像教练脑子）
"""



"""
python -m aegis.models.value.train_value_xgb \
  --value data/processed/value_samples_h180.parquet \
  --splits_dir data/splits \
  --out_dir artifacts/value_xgb_h180
"""

"""
python -m aegis.models.value.train_value_xgb \
  --value data/processed/value_samples_v1_h180.parquet \
  --splits_dir data/splits \
  --out_dir artifacts/value_xgb_v1_h180
"""

import argparse, os
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

FEW_SHOT_NS = [0, 25, 50, 100, 200, 500, 1000]

def read_ids(path: str) -> list[int]:
    with open(path, "r") as f:
        return [int(x.strip()) for x in f if x.strip()]

def load_split_df(value_path: str, match_ids: list[int]) -> pl.DataFrame:
    ids = pl.DataFrame({"match_id": match_ids})
    df = (
        pl.scan_parquet(value_path)
        .join(ids.lazy(), on="match_id", how="inner")
        .collect()
    )
    return df

def add_match_weights(df: pl.DataFrame) -> pl.DataFrame:
    # weight each match equally (avoid long matches dominating)
    counts = df.group_by("match_id").agg(pl.len().alias("n_rows"))
    df = df.join(counts, on="match_id", how="left")
    df = df.with_columns((1.0 / pl.col("n_rows")).alias("w_match")).drop("n_rows")
    return df

def to_xyw(df: pl.DataFrame):
    meta_cols = {"match_id", "patch"}  # keep t as feature
    label_cols = {"y_win", "y_dgold_adv", "y_dxp_adv"}
    cols = df.columns

    feature_cols = [c for c in cols if c not in meta_cols and c not in label_cols and c != "w_match"]
    X = df.select(feature_cols).to_pandas()
    y_win = df["y_win"].to_numpy()
    y_dg = df["y_dgold_adv"].to_numpy()
    y_dx = df["y_dxp_adv"].to_numpy()
    w = df["w_match"].to_numpy() if "w_match" in df.columns else None
    return feature_cols, X, y_win, y_dg, y_dx, w

def eval_metrics(name: str, y_win, p_win, y_dg, pred_dg, y_dx, pred_dx):
    auc = roc_auc_score(y_win, p_win)
    ll = log_loss(y_win, np.clip(p_win, 1e-6, 1-1e-6))

    mse_dg = mean_squared_error(y_dg, pred_dg)
    rmse_dg = float(np.sqrt(mse_dg))
    mae_dg = mean_absolute_error(y_dg, pred_dg)

    mse_dx = mean_squared_error(y_dx, pred_dx)
    rmse_dx = float(np.sqrt(mse_dx))
    mae_dx = mean_absolute_error(y_dx, pred_dx)

    print(f"\n== {name} ==")
    print(f"win:  AUC={auc:.4f}  LogLoss={ll:.4f}")
    print(f"dGoldAdv@3m: RMSE={rmse_dg:.2f}  MAE={mae_dg:.2f}")
    print(f"dXPAdv@3m:   RMSE={rmse_dx:.2f}  MAE={mae_dx:.2f}")

def calibrate_sigmoid(y_true: np.ndarray, p: np.ndarray):
    # Platt scaling on logits
    p = np.clip(p, 1e-6, 1-1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(logit, y_true)
    return lr

def apply_sigmoid_calibration(lr: LogisticRegression, p: np.ndarray):
    p = np.clip(p, 1e-6, 1-1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    return lr.predict_proba(logit)[:, 1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", default="data/processed/value_samples_h180.parquet")
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--out_dir", default="artifacts/value_xgb_h180")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train_rows", type=int, default=0, help="0 means no cap")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    train_ids = read_ids(os.path.join(args.splits_dir, "train.txt"))
    val_ids   = read_ids(os.path.join(args.splits_dir, "val.txt"))
    test_ids  = read_ids(os.path.join(args.splits_dir, "test.txt"))

    df_train = add_match_weights(load_split_df(args.value, train_ids))
    df_val   = add_match_weights(load_split_df(args.value, val_ids))
    df_test  = add_match_weights(load_split_df(args.value, test_ids))

    # optional cap for faster iteration
    if args.max_train_rows and len(df_train) > args.max_train_rows:
        df_train = df_train.sample(n=args.max_train_rows, seed=args.seed)

    feature_cols, X_tr, ywin_tr, ydg_tr, ydx_tr, w_tr = to_xyw(df_train)
    _,           X_va, ywin_va, ydg_va, ydx_va, w_va = to_xyw(df_val)
    _,           X_te, ywin_te, ydg_te, ydx_te, w_te = to_xyw(df_test)

    # --- Models
    clf = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=50,   # ✅ 加这一行
        random_state=args.seed,
        tree_method="hist",
    )

    reg_dg = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=50,   # ✅ 加这一行
        random_state=args.seed,
        tree_method="hist",
    )

    reg_dx = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=50,   # ✅ 加这一行
        random_state=args.seed,
        tree_method="hist",
    )

    # --- Fit with early stopping on val
    clf.fit(X_tr, ywin_tr, sample_weight=w_tr, eval_set=[(X_va, ywin_va)], verbose=False)
    reg_dg.fit(X_tr, ydg_tr, sample_weight=w_tr, eval_set=[(X_va, ydg_va)], verbose=False)
    reg_dx.fit(X_tr, ydx_tr, sample_weight=w_tr, eval_set=[(X_va, ydx_va)], verbose=False)

    # --- Evaluate (zero-shot)
    p_te = clf.predict_proba(X_te)[:, 1]
    dg_te = reg_dg.predict(X_te)
    dx_te = reg_dx.predict(X_te)
    eval_metrics("TEST (patch59) zero-shot", ywin_te, p_te, ydg_te, dg_te, ydx_te, dx_te)

    # Save models + feature list
    clf.save_model(os.path.join(args.out_dir, "win_xgb.json"))
    reg_dg.save_model(os.path.join(args.out_dir, "dgold_xgb.json"))
    reg_dx.save_model(os.path.join(args.out_dir, "dxp_xgb.json"))
    with open(os.path.join(args.out_dir, "features.txt"), "w") as f:
        for c in feature_cols:
            f.write(c + "\n")

    # --- Few-shot adaptation curve on test patch (calibration, not retraining)
    print("\n==== Few-shot adaptation (calibration) ====")
    for n in FEW_SHOT_NS:
        if n == 0:
            # already printed above for full test; also show same metrics on eval_0 == test
            continue

        calib_ids = read_ids(os.path.join(args.splits_dir, f"test_calib_{n}.txt"))
        eval_ids  = read_ids(os.path.join(args.splits_dir, f"test_eval_{n}.txt"))

        df_cal = add_match_weights(load_split_df(args.value, calib_ids))
        df_ev  = add_match_weights(load_split_df(args.value, eval_ids))

        _, X_cal, ywin_cal, ydg_cal, ydx_cal, _ = to_xyw(df_cal)
        _, X_ev,  ywin_ev,  ydg_ev,  ydx_ev,  _ = to_xyw(df_ev)

        # base predictions
        p_cal = clf.predict_proba(X_cal)[:, 1]
        p_ev  = clf.predict_proba(X_ev)[:, 1]
        dg_cal = reg_dg.predict(X_cal); dg_ev = reg_dg.predict(X_ev)
        dx_cal = reg_dx.predict(X_cal); dx_ev = reg_dx.predict(X_ev)

        # classification calibration: sigmoid
        lr = calibrate_sigmoid(ywin_cal, p_cal)
        p_ev_c = apply_sigmoid_calibration(lr, p_ev)

        # regression bias correction
        dg_bias = float(np.mean(ydg_cal - dg_cal))
        dx_bias = float(np.mean(ydx_cal - dx_cal))
        dg_ev_c = dg_ev + dg_bias
        dx_ev_c = dx_ev + dx_bias

        eval_metrics(f"TEST eval (N={n}) after calib", ywin_ev, p_ev_c, ydg_ev, dg_ev_c, ydx_ev, dx_ev_c)

if __name__ == "__main__":
    main()
