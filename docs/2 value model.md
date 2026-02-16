跑run_v1.sh



✅ v1 输出没问题，提升很明显
* AUC 0.7807 vs v0 0.7681
* Δadv RMSE 大幅下降（dGold 2761→2280，dXP 5034→4271）这说明 items/wards/skirmish 特征确实让 value 更“行动敏感”。





对，**Phase 1 的 Value Model 先不用 `events`** ✅（先用 `state_player_minute + state_team_minute + matches` 做一个强 baseline 打分器；`events` 主要是 Phase 1 后半（next-item / ward policy）和 Phase 2（planner/world model）的燃料。）

下面给你两份**复制就能跑**的脚本：

* `aegis/etl/make_value_samples.py`：生成 `value_samples.parquet`（H=180s 默认）
* `aegis/models/value/train_value_xgb.py`：XGBoost baseline（win + Δgold_adv + Δxp_adv）+ zero-shot test(7.40) + few-shot calibration curve

---

# 1) 生成 value_samples.parquet（Polars, H=180s）

把下面保存为：`aegis/etl/make_value_samples.py`

```python
import argparse
import os
import polars as pl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches_glob", default="data/processed/matches/*.parquet")
    ap.add_argument("--team_glob", default="data/processed/state_team_minute/*.parquet")
    ap.add_argument("--player_glob", default="data/processed/state_player_minute/*.parquet")
    ap.add_argument("--out", default="data/processed/value_samples_h180.parquet")
    ap.add_argument("--horizon_sec", type=int, default=180)  # 3 minutes
    ap.add_argument("--compression", default="zstd")
    args = ap.parse_args()

    H = args.horizon_sec
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    matches = (
        pl.scan_parquet(args.matches_glob)
        .select(["match_id", "radiant_win"])
    )

    team = (
        pl.scan_parquet(args.team_glob)
        .select(["match_id", "patch", "t", "radiant_gold_adv", "radiant_xp_adv"])
    )

    player = (
        pl.scan_parquet(args.player_glob)
        .select(["match_id", "patch", "t", "team", "gold", "xp", "lh", "dn"])
    )

    # --- aggregate per team per minute
    agg = (
        player.group_by(["match_id", "patch", "t", "team"])
        .agg([
            pl.sum("gold").alias("gold_sum"),
            pl.sum("xp").alias("xp_sum"),
            pl.sum("lh").alias("lh_sum"),
            pl.sum("dn").alias("dn_sum"),
            pl.mean("gold").alias("gold_mean"),
            pl.mean("xp").alias("xp_mean"),
            pl.max("gold").alias("gold_max"),
            pl.max("xp").alias("xp_max"),
        ])
    )

    rad = (
        agg.filter(pl.col("team") == 0)
        .drop("team")
        .rename({
            "gold_sum": "rad_gold_sum",
            "xp_sum": "rad_xp_sum",
            "lh_sum": "rad_lh_sum",
            "dn_sum": "rad_dn_sum",
            "gold_mean": "rad_gold_mean",
            "xp_mean": "rad_xp_mean",
            "gold_max": "rad_gold_max",
            "xp_max": "rad_xp_max",
        })
    )

    dire = (
        agg.filter(pl.col("team") == 1)
        .drop("team")
        .rename({
            "gold_sum": "dire_gold_sum",
            "xp_sum": "dire_xp_sum",
            "lh_sum": "dire_lh_sum",
            "dn_sum": "dire_dn_sum",
            "gold_mean": "dire_gold_mean",
            "xp_mean": "dire_xp_mean",
            "gold_max": "dire_gold_max",
            "xp_max": "dire_xp_max",
        })
    )

    feat = rad.join(dire, on=["match_id", "patch", "t"], how="inner")

    # diffs (radiant - dire)
    feat = feat.with_columns([
        (pl.col("rad_gold_sum") - pl.col("dire_gold_sum")).alias("gold_sum_diff"),
        (pl.col("rad_xp_sum") - pl.col("dire_xp_sum")).alias("xp_sum_diff"),
        (pl.col("rad_lh_sum") - pl.col("dire_lh_sum")).alias("lh_sum_diff"),
        (pl.col("rad_dn_sum") - pl.col("dire_dn_sum")).alias("dn_sum_diff"),
        (pl.col("rad_gold_mean") - pl.col("dire_gold_mean")).alias("gold_mean_diff"),
        (pl.col("rad_xp_mean") - pl.col("dire_xp_mean")).alias("xp_mean_diff"),
        (pl.col("rad_gold_max") - pl.col("dire_gold_max")).alias("gold_max_diff"),
        (pl.col("rad_xp_max") - pl.col("dire_xp_max")).alias("xp_max_diff"),
        (pl.col("t") / 60.0).alias("t_min"),
    ])

    # join team advantages (current)
    feat = feat.join(team, on=["match_id", "patch", "t"], how="inner")

    # create future advantages table aligned to current t
    future = (
        team.select(["match_id", "patch", "t", "radiant_gold_adv", "radiant_xp_adv"])
        .with_columns((pl.col("t") - H).alias("t"))  # shift back so future aligns to current
        .rename({
            "radiant_gold_adv": "radiant_gold_adv_f",
            "radiant_xp_adv": "radiant_xp_adv_f",
        })
    )

    df = feat.join(future, on=["match_id", "patch", "t"], how="left")

    # keep rows where future exists and current adv exists
    df = df.drop_nulls(["radiant_gold_adv", "radiant_xp_adv", "radiant_gold_adv_f", "radiant_xp_adv_f"])

    # labels
    df = df.with_columns([
        (pl.col("radiant_gold_adv_f") - pl.col("radiant_gold_adv")).alias("y_dgold_adv"),
        (pl.col("radiant_xp_adv_f") - pl.col("radiant_xp_adv")).alias("y_dxp_adv"),
    ]).drop(["radiant_gold_adv_f", "radiant_xp_adv_f"])

    # join win label
    df = df.join(matches, on="match_id", how="left").drop_nulls(["radiant_win"])
    df = df.with_columns(pl.col("radiant_win").cast(pl.Int8).alias("y_win")).drop("radiant_win")

    # NOTE: do NOT include patch as a model feature by default; keep it for slicing
    # Write parquet (streaming if available)
    try:
        df.sink_parquet(args.out, compression=args.compression)
    except Exception:
        df.collect(streaming=True).write_parquet(args.out, compression=args.compression)

    print(f"Wrote: {args.out} (horizon={H}s)")

if __name__ == "__main__":
    main()
```

运行：

```bash
python -m aegis.etl.make_value_samples \
  --out data/processed/value_samples_h180.parquet \
  --horizon_sec 180
```

---

# 2) 训练 Value Model（XGBoost baseline + zero-shot + few-shot curve）

先装依赖：

```bash
pip install xgboost scikit-learn pandas numpy
```

把下面保存为：`aegis/models/value/train_value_xgb.py`

```python
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
    rmse_dg = mean_squared_error(y_dg, pred_dg, squared=False)
    mae_dg = mean_absolute_error(y_dg, pred_dg)
    rmse_dx = mean_squared_error(y_dx, pred_dx, squared=False)
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
```

运行：

```bash
python -m aegis.models.value.train_value_xgb \
  --value data/processed/value_samples_h180.parquet \
  --splits_dir data/splits \
  --out_dir artifacts/value_xgb_h180
```

> 如果你想先快跑验证 pipeline：加 `--max_train_rows 300000`。

---

# 3) 为什么 Phase 1 的 Value 先不吃 events？

* **Value baseline** 只需要“当前经济/经验/补刀/优势曲线”就能很强（尤其 Δadv 预测）。
* `events` 的价值主要在下一步：

  * **Next-item policy**：需要 purchase events 的细粒度序列
  * **Ward policy**：需要 ward 坐标/存活（place/left）
  * **Planner/world model**：需要统一事件流来做 plan tokens / rollout / drift control

你现在先把 Value 打分器跑通，下一步我们就可以做 **Generate-and-Score** 的第一个 demo（即便 policy 先用简单的 top-k 规则生成候选也行）。

---

你先把 `value_samples_h180.parquet` 生成出来，跑一下训练脚本，把输出的 AUC/RMSE 贴我一下。下一条我就把 **Generate-and-Score MVP（候选装备/眼位生成 + value 打分选优）** 的 PRD/代码骨架继续给你接上。
