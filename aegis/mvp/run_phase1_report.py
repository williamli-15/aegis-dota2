"""
python -m aegis.mvp.run_phase1_report --feasible
"""

import argparse, os, json
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, mean_absolute_error

def read_ids(path: str):
    with open(path, "r") as f:
        return [int(x.strip()) for x in f if x.strip()]

def load_booster(path: str) -> xgb.Booster:
    b = xgb.Booster()
    b.load_model(path)
    return b

def safe_select(lf: pl.LazyFrame, cols: list[str]) -> pl.LazyFrame:
    schema = set(lf.collect_schema().names())
    exprs = []
    for c in cols:
        if c in schema:
            exprs.append(pl.col(c).fill_null(0).cast(pl.Float32).alias(c))
        else:
            exprs.append(pl.lit(0.0).cast(pl.Float32).alias(c))
    return lf.select(exprs)

def eval_value(model_dir: str, value_parquet: str, splits_dir: str, split_name="test"):
    feat_path = os.path.join(model_dir, "features.txt")
    win_path = os.path.join(model_dir, "win_xgb.json")
    dg_path  = os.path.join(model_dir, "dgold_xgb.json")
    dx_path  = os.path.join(model_dir, "dxp_xgb.json")

    feature_cols = [x.strip() for x in open(feat_path) if x.strip()]
    win_b = load_booster(win_path)
    dg_b  = load_booster(dg_path)
    dx_b  = load_booster(dx_path)

    ids = pl.DataFrame({"match_id": read_ids(os.path.join(splits_dir, f"{split_name}.txt"))})

    lf = pl.scan_parquet(value_parquet).join(ids.lazy(), on="match_id", how="inner")
    lf = lf.select(["y_win","y_dgold_adv","y_dxp_adv"] + feature_cols)
    df = lf.collect()

    y_win = df["y_win"].to_numpy()
    y_dg  = df["y_dgold_adv"].to_numpy()
    y_dx  = df["y_dxp_adv"].to_numpy()

    X = df.select(feature_cols).to_numpy().astype(np.float32, copy=False)
    dmat = xgb.DMatrix(X, feature_names=feature_cols)

    p_win = np.asarray(win_b.predict(dmat)).reshape(-1)
    p_win = np.clip(p_win, 1e-6, 1-1e-6)
    pred_dg = np.asarray(dg_b.predict(dmat)).reshape(-1)
    pred_dx = np.asarray(dx_b.predict(dmat)).reshape(-1)

    auc = roc_auc_score(y_win, p_win)
    ll = log_loss(y_win, p_win)

    rmse_dg = float(np.sqrt(mean_squared_error(y_dg, pred_dg)))
    mae_dg  = float(mean_absolute_error(y_dg, pred_dg))
    rmse_dx = float(np.sqrt(mean_squared_error(y_dx, pred_dx)))
    mae_dx  = float(mean_absolute_error(y_dx, pred_dx))

    return {
        "AUC": auc, "LogLoss": ll,
        "RMSE_dGold": rmse_dg, "MAE_dGold": mae_dg,
        "RMSE_dXP": rmse_dx, "MAE_dXP": mae_dx,
        "n_rows": int(df.height)
    }

def eval_next_item_policy(policy_dir: str, samples_parquet: str, splits_dir: str, topk=20):
    booster = xgb.Booster()
    booster.load_model(os.path.join(policy_dir, "next_item_xgb.json"))

    with open(os.path.join(policy_dir, "feature_cols.json"), "r") as f:
        pol_cols = json.load(f)
    with open(os.path.join(policy_dir, "id_to_label.json"), "r") as f:
        id_to_label = {int(k): v for k, v in json.load(f).items()}
    label_to_id = {v:k for k,v in id_to_label.items()}
    C = len(id_to_label)

    ids = pl.DataFrame({"match_id": read_ids(os.path.join(splits_dir, "test.txt"))})
    df = pl.scan_parquet(samples_parquet).join(ids.lazy(), on="match_id", how="inner").collect()

    # drop unseen labels
    df = df.filter(pl.col("y_next_item").is_in(list(label_to_id.keys())))
    y = np.array([label_to_id[v] for v in df["y_next_item"].to_list()], dtype=np.int32)

    X = df.select([pl.col(c).fill_null(0).cast(pl.Float32) for c in pol_cols]).to_numpy()
    dmat = xgb.DMatrix(X, feature_names=pol_cols)
    proba = np.asarray(booster.predict(dmat))
    if proba.ndim == 1:
        proba = proba.reshape(-1, C)

    top1 = float((proba.argmax(axis=1) == y).mean())
    def topk_acc(k):
        k = min(k, proba.shape[1])
        idx = np.argpartition(-proba, kth=k-1, axis=1)[:, :k]
        return float((idx == y[:,None]).any(axis=1).mean())

    return {
        "Top1": top1,
        "Top5": topk_acc(5),
        f"Top{topk}": topk_acc(topk),
        "n_rows": int(df.height),
        "classes": int(C),
    }

def load_item_cost(constants_path: str) -> dict[str,int]:
    def _cost(v):
        c = v.get("cost", 0)
        return int(c) if c is not None else 0
    if not os.path.exists(constants_path):
        return {}
    with open(constants_path, "r") as f:
        data = json.load(f)
    return {k:_cost(v) for k,v in data.items()}

def load_item_t50(cache_path: str) -> dict[str,float]:
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r") as f:
        return {k: float(v) for k,v in json.load(f).items()}

def eval_generate_and_score(samples_parquet: str, policy_dir: str, value_dir: str, value_features: str,
                            splits_dir: str, n_eval=5000, topk=20, seed=42,
                            feasible=False, min_item_cost=0, time_band_sec=0, gold_buffer=800.0,
                            item_cost_path="data/processed/opendota_items_constants.json",
                            item_t50_path="data/processed/item_t50_cache.json"):

    np.random.seed(seed)

    # load test sample rows
    ids = pl.DataFrame({"match_id": read_ids(os.path.join(splits_dir, "test.txt"))})
    df = pl.scan_parquet(samples_parquet).join(ids.lazy(), on="match_id", how="inner").collect()
    if df.height > n_eval:
        df = df.sample(n=n_eval, seed=seed)

    # policy
    policy = xgb.Booster()
    policy.load_model(os.path.join(policy_dir, "next_item_xgb.json"))
    with open(os.path.join(policy_dir, "feature_cols.json"), "r") as f:
        pol_cols = json.load(f)
    with open(os.path.join(policy_dir, "id_to_label.json"), "r") as f:
        id_to_label = {int(k): v for k, v in json.load(f).items()}
    label_to_id = {v:k for k,v in id_to_label.items()}
    Cexp = len(id_to_label)

    df = df.filter(pl.col("y_next_item").is_in(list(label_to_id.keys())))
    if df.height == 0:
        raise RuntimeError("No rows left after label filtering")

    # value win head
    win_b = xgb.Booster()
    win_b.load_model(os.path.join(value_dir, "win_xgb.json"))
    val_cols = [x.strip() for x in open(value_features) if x.strip()]
    val_idx = {c:i for i,c in enumerate(val_cols)}

    # proba
    X_pol = df.select([pl.col(c).fill_null(0).cast(pl.Float32) for c in pol_cols]).to_numpy()
    proba = np.asarray(policy.inplace_predict(X_pol))
    if proba.ndim == 1:
        proba = proba.reshape(-1, Cexp)

    y_true = np.array([label_to_id[v] for v in df["y_next_item"].to_list()], dtype=np.int32)
    k = min(topk, proba.shape[1])
    topk_idx = np.argpartition(-proba, kth=k-1, axis=1)[:, :k]
    policy_hit = float((topk_idx == y_true[:,None]).any(axis=1).mean())

    # base value matrix
    X_base = df.select([pl.col(c).fill_null(0).cast(pl.Float32) for c in val_cols]).to_numpy().astype(np.float32, copy=False)
    teams = df["team"].to_numpy().astype(np.int32, copy=False)
    t_arr = df["t"].to_numpy().astype(np.int32, copy=False)
    true_items = df["y_next_item"].to_list()

    rad_gold_max = df["rad_gold_max"].to_numpy().astype(np.float32, copy=False)
    dire_gold_max = df["dire_gold_max"].to_numpy().astype(np.float32, copy=False)
    gold_max_team = np.where(teams==0, rad_gold_max, dire_gold_max)

    p0 = np.asarray(win_b.inplace_predict(X_base)).reshape(-1)
    base_tw = np.where(teams==0, p0, 1.0-p0)

    item_cost = load_item_cost(item_cost_path) if (feasible or min_item_cost>0) else {}
    item_t50 = load_item_t50(item_t50_path) if time_band_sec>0 else {}

    def apply_item(xrow: np.ndarray, item: str, team: int):
        rad = val_idx.get(f"rad_has_{item}")
        dire = val_idx.get(f"dire_has_{item}")
        diff = val_idx.get(f"diff_has_{item}")
        if rad is None and dire is None and diff is None:
            return
        if team==0:
            if rad is not None: xrow[rad]=1.0
        else:
            if dire is not None: xrow[dire]=1.0
        if diff is not None:
            r = xrow[rad] if rad is not None else 0.0
            d = xrow[dire] if dire is not None else 0.0
            xrow[diff] = r-d

    chosen_hit = 0
    n_true_ok = 0
    uplift_true = []
    uplift_base = []

    for i in range(df.height):
        team = int(teams[i])
        t_used = int(t_arr[i])
        gmax = float(gold_max_team[i])
        true_it = true_items[i]

        cand = [id_to_label[int(cid)] for cid in topk_idx[i]]
        seen=set(); cand=[x for x in cand if not (x in seen or seen.add(x))]
        if true_it not in cand:
            cand.append(true_it)

        # filter
        filt=[]
        for it in cand:
            cost = item_cost.get(it, 0)
            if min_item_cost>0 and cost < min_item_cost:
                continue
            if time_band_sec>0:
                t50 = item_t50.get(it)
                if t50 is not None and abs(t_used - t50) > time_band_sec:
                    continue
            if feasible and cost>0 and cost > (gmax + gold_buffer):
                continue
            filt.append(it)

        if len(filt)==0:
            uplift_base.append(0.0)
            continue

        Xcand = np.repeat(X_base[i:i+1], len(filt), axis=0)
        for j,it in enumerate(filt):
            apply_item(Xcand[j], it, team)

        p = np.asarray(win_b.inplace_predict(Xcand)).reshape(-1)
        tw = p if team==0 else (1.0-p)

        best_j = int(np.argmax(tw))
        best_tw = float(tw[best_j])
        best_it = filt[best_j]
        uplift_base.append(best_tw - float(base_tw[i]))

        if true_it in filt:
            n_true_ok += 1
            tj = filt.index(true_it)
            uplift_true.append(best_tw - float(tw[tj]))
            if best_it == true_it:
                chosen_hit += 1

    out = {
        "policy_hit@k": policy_hit,
        "avg_uplift_vs_base": float(np.mean(uplift_base)),
        "n_eval": int(df.height),
        "k": int(k),
    }
    if n_true_ok>0:
        out.update({
            "gs_hit@1_true_ok": chosen_hit / n_true_ok,
            "avg_uplift_vs_true_true_ok": float(np.mean(uplift_true)),
            "n_true_ok": int(n_true_ok),
        })
    else:
        out.update({"gs_hit@1_true_ok": None, "avg_uplift_vs_true_true_ok": None, "n_true_ok": 0})
    return out

def md_table(rows, headers):
    def fmt(v):
        if v is None: return "—"
        if isinstance(v, float): return f"{v:.4f}"
        return str(v)
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"]*len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(h)) for h in headers) + " |")
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--out", default="artifacts/phase1_report.md")

    ap.add_argument("--value0_parquet", default="data/processed/value_samples_h180.parquet")
    ap.add_argument("--value0_model_dir", default="artifacts/value_xgb_h180")
    ap.add_argument("--value1_parquet", default="data/processed/value_samples_v1_h180.parquet")
    ap.add_argument("--value1_model_dir", default="artifacts/value_xgb_v1_h180")

    ap.add_argument("--next_item_samples", default="data/processed/next_item_samples.parquet")
    ap.add_argument("--policy_dir", default="artifacts/next_item_policy_xgb")

    ap.add_argument("--n_eval", type=int, default=5000)
    ap.add_argument("--topk", type=int, default=20)

    ap.add_argument("--feasible", action="store_true")
    ap.add_argument("--min_item_cost", type=int, default=2000)
    ap.add_argument("--time_band_sec", type=int, default=900)
    ap.add_argument("--gold_buffer", type=float, default=800.0)

    ap.add_argument("--item_cost_path", default="data/processed/opendota_items_constants.json")
    ap.add_argument("--item_t50_path", default="data/processed/item_t50_cache.json")

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    v0 = eval_value(args.value0_model_dir, args.value0_parquet, args.splits_dir, "test")
    v1 = eval_value(args.value1_model_dir, args.value1_parquet, args.splits_dir, "test")

    pol = eval_next_item_policy(args.policy_dir, args.next_item_samples, args.splits_dir, topk=args.topk)

    gs_magic = eval_generate_and_score(
        args.next_item_samples, args.policy_dir, args.value1_model_dir,
        os.path.join(args.value1_model_dir, "features.txt"),
        args.splits_dir, n_eval=args.n_eval, topk=args.topk,
        feasible=False, min_item_cost=0, time_band_sec=0,
        item_cost_path=args.item_cost_path, item_t50_path=args.item_t50_path
    )

    gs_feas = eval_generate_and_score(
        args.next_item_samples, args.policy_dir, args.value1_model_dir,
        os.path.join(args.value1_model_dir, "features.txt"),
        args.splits_dir, n_eval=args.n_eval, topk=args.topk,
        feasible=args.feasible, min_item_cost=args.min_item_cost, time_band_sec=args.time_band_sec,
        gold_buffer=args.gold_buffer,
        item_cost_path=args.item_cost_path, item_t50_path=args.item_t50_path
    )

    lines = []
    lines.append("# Phase 1 Report\n")
    lines.append("## Value Model (TEST patch)\n")
    lines.append(md_table([
        {"Model":"Value v0", **v0},
        {"Model":"Value v1", **v1},
    ], ["Model","AUC","LogLoss","RMSE_dGold","MAE_dGold","RMSE_dXP","MAE_dXP","n_rows"]))
    lines.append("\n## Next-item Policy (TEST)\n")
    lines.append(md_table([{
        "Top1": pol["Top1"], "Top5": pol["Top5"], f"Top{args.topk}": pol[f"Top{args.topk}"],
        "classes": pol["classes"], "n_rows": pol["n_rows"]
    }], ["Top1","Top5",f"Top{args.topk}","classes","n_rows"]))
    lines.append("\n## Generate-and-Score (TEST)\n")
    lines.append("### Magic toggle\n")
    lines.append(md_table([gs_magic], ["policy_hit@k","k","avg_uplift_vs_base","gs_hit@1_true_ok","avg_uplift_vs_true_true_ok","n_true_ok","n_eval"]))
    lines.append("\n### Feasible/time-band\n")
    lines.append(md_table([gs_feas], ["policy_hit@k","k","avg_uplift_vs_base","gs_hit@1_true_ok","avg_uplift_vs_true_true_ok","n_true_ok","n_eval"]))

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("Wrote report:", args.out)

if __name__ == "__main__":
    main()
