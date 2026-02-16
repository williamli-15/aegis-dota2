"""
python -m aegis.mvp.score_actions_v2 \
  --match_id 8607168614 --t 900 --team 0 --mode both --topk 15 \
  --feasible --fetch_constants --min_item_cost 2000 --time_band_sec 900
"""
import argparse, os, json
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

def load_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]

def load_booster(path: str) -> xgb.Booster:
    b = xgb.Booster()
    b.load_model(path)
    return b

def dict_to_X(feat: Dict[str, float], feature_cols: List[str]) -> pd.DataFrame:
    data = {c: [feat.get(c, 0.0)] for c in feature_cols}
    return pd.DataFrame(data)

def predict_all(win_b: xgb.Booster, dg_b: xgb.Booster, dx_b: xgb.Booster,
                X: pd.DataFrame) -> Tuple[float, float, float]:
    dmat = xgb.DMatrix(X, feature_names=list(X.columns))
    p_win = float(win_b.predict(dmat)[0])
    p_dg = float(dg_b.predict(dmat)[0])
    p_dx = float(dx_b.predict(dmat)[0])
    return p_win, p_dg, p_dx

def team_view(team: int, p_win: float, p_dg: float, p_dx: float) -> Tuple[float, float, float]:
    # value model is radiant-centric; convert to team-centric view
    if team == 0:
        return p_win, p_dg, p_dx
    return 1.0 - p_win, -p_dg, -p_dx

def load_row(value_parquet: str, match_id: int, t: int) -> Dict[str, float]:
    df = (
        pl.scan_parquet(value_parquet)
        .filter((pl.col("match_id") == match_id) & (pl.col("t") == t))
        .collect()
    )
    if df.height == 0:
        ts = (
            pl.scan_parquet(value_parquet)
            .filter(pl.col("match_id") == match_id)
            .select(["t"])
            .collect()
        )
        if ts.height == 0:
            raise ValueError(f"No rows for match_id={match_id}")
        arr = ts["t"].to_numpy()
        t_closest = int(arr[np.argmin(np.abs(arr - t))])
        df = (
            pl.scan_parquet(value_parquet)
            .filter((pl.col("match_id") == match_id) & (pl.col("t") == t_closest))
            .collect()
        )
        print(f"[warn] t={t} not found; using closest t={t_closest}")
    row = df.row(0, named=True)
    out = {}
    for k, v in row.items():
        if v is None:
            continue
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out

def load_item_cost(constants_path: str, fetch: bool=False) -> Dict[str, int]:
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

def compute_item_t50(events_glob: str, vocab: List[str], out_cache: str) -> Dict[str, float]:
    # cache to avoid re-scanning events every run
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

def already_has_item(feat: Dict[str, float], item: str, team: int) -> bool:
    c = ("rad_" if team == 0 else "dire_") + f"has_{item}"
    return feat.get(c, 0.0) >= 0.5

def set_item(feat: Dict[str, float], item: str, team: int):
    rad_col = f"rad_has_{item}"
    dire_col = f"dire_has_{item}"
    diff_col = f"diff_has_{item}"

    feat.setdefault(rad_col, 0.0)
    feat.setdefault(dire_col, 0.0)

    if team == 0:
        feat[rad_col] = 1.0
    else:
        feat[dire_col] = 1.0

    if diff_col in feat:
        feat[diff_col] = feat.get(rad_col, 0.0) - feat.get(dire_col, 0.0)

def apply_cost_adjust(feat: Dict[str, float], team: int, cost: int):
    # optional: deduct gold approx (keeps state more "feasible")
    if cost <= 0:
        return

    if team == 0:
        rs, rm, rx = "rad_gold_sum", "rad_gold_mean", "rad_gold_max"
        ds, dm, dx = "dire_gold_sum", "dire_gold_mean", "dire_gold_max"
    else:
        ds, dm, dx = "rad_gold_sum", "rad_gold_mean", "rad_gold_max"
        rs, rm, rx = "dire_gold_sum", "dire_gold_mean", "dire_gold_max"

    # clamp >= 0
    def dec(k, v):
        if k in feat:
            feat[k] = max(0.0, feat[k] - v)

    dec(rs, float(cost))
    dec(rm, float(cost) / 5.0)
    dec(rx, float(cost))

    # update diff if present (radiant - dire)
    if "gold_sum_diff" in feat and "rad_gold_sum" in feat and "dire_gold_sum" in feat:
        feat["gold_sum_diff"] = feat["rad_gold_sum"] - feat["dire_gold_sum"]
    if "gold_mean_diff" in feat and "rad_gold_mean" in feat and "dire_gold_mean" in feat:
        feat["gold_mean_diff"] = feat["rad_gold_mean"] - feat["dire_gold_mean"]
    if "gold_max_diff" in feat and "rad_gold_max" in feat and "dire_gold_max" in feat:
        feat["gold_max_diff"] = feat["rad_gold_max"] - feat["dire_gold_max"]

def add_ward(feat: Dict[str, float], team: int, kind: str):
    pref = "rad_" if team == 0 else "dire_"
    other = "dire_" if team == 0 else "rad_"

    if kind == "obs":
        bases = ["active_obs","obs_place_1m","obs_place_5m"]
    else:
        bases = ["active_sen","sen_place_1m","sen_place_5m"]

    for b in bases:
        feat[pref + b] = feat.get(pref + b, 0.0) + 1.0
        # update diff if exists
        diff = "diff_" + b
        if diff in feat:
            feat.setdefault(other + b, 0.0)
            feat["diff_" + b] = feat.get("rad_" + b, 0.0) - feat.get("dire_" + b, 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value_parquet", default="data/processed/value_samples_v1_h180.parquet")
    ap.add_argument("--models_dir", default="artifacts/value_xgb_v1_h180")
    ap.add_argument("--item_vocab", default="data/processed/item_vocab_v1.txt")
    ap.add_argument("--events_glob", default="data/processed/events/*.parquet")
    ap.add_argument("--item_meta_cache", default="data/processed/item_t50_cache.json")

    ap.add_argument("--match_id", type=int, required=True)
    ap.add_argument("--t", type=int, default=900)
    ap.add_argument("--team", type=int, default=0, choices=[0,1])
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--mode", default="both", choices=["items","wards","both"])

    ap.add_argument("--feasible", action="store_true", help="filter items by cost<=team_gold_max (needs constants)")
    ap.add_argument("--min_item_cost", type=int, default=0)
    ap.add_argument("--fetch_constants", action="store_true")
    ap.add_argument("--item_constants", default="data/processed/opendota_items_constants.json")

    ap.add_argument("--time_band_sec", type=int, default=900, help="|t - item_t50| <= band; 0 disables")
    ap.add_argument("--cost_adjust", action="store_true", help="optionally deduct gold features (approx)")

    args = ap.parse_args()

    features_path = os.path.join(args.models_dir, "features.txt")
    win_path = os.path.join(args.models_dir, "win_xgb.json")
    dg_path  = os.path.join(args.models_dir, "dgold_xgb.json")
    dx_path  = os.path.join(args.models_dir, "dxp_xgb.json")

    feature_cols = load_lines(features_path)
    vocab = load_lines(args.item_vocab)

    win_b = load_booster(win_path)
    dg_b  = load_booster(dg_path)
    dx_b  = load_booster(dx_path)

    base = load_row(args.value_parquet, args.match_id, args.t)
    t_used = int(base.get("t", args.t))

    X0 = dict_to_X(base, feature_cols)
    p_win0, p_dg0, p_dx0 = predict_all(win_b, dg_b, dx_b, X0)
    tv0, tdg0, tdx0 = team_view(args.team, p_win0, p_dg0, p_dx0)

    print(f"\nBase @ match={args.match_id} t={t_used} team={args.team} (0=Rad 1=Dire)")
    print(f"  team_win_prob={tv0:.4f}   team_dGoldAdv={tdg0:.1f}   team_dXPAdv={tdx0:.1f}")

    item_cost = {}
    if args.feasible or args.min_item_cost > 0:
        item_cost = load_item_cost(args.item_constants, fetch=args.fetch_constants)

    item_t50 = {}
    if args.time_band_sec > 0:
        item_t50 = compute_item_t50(args.events_glob, vocab, args.item_meta_cache)

    # available gold proxy
    gold_max = base.get("rad_gold_max", None) if args.team == 0 else base.get("dire_gold_max", None)
    if gold_max is None:
        gold_max = 0.0

    results = []

    # ---- items
    if args.mode in ("items","both"):
        for it in vocab:
            if already_has_item(base, it, args.team):
                continue

            cost = item_cost.get(it, 0)

            if args.min_item_cost > 0 and cost < args.min_item_cost:
                continue

            if args.feasible and cost > 0 and gold_max > 0 and cost > gold_max:
                continue

            if args.time_band_sec > 0 and it in item_t50:
                if abs(t_used - item_t50[it]) > args.time_band_sec:
                    continue

            feat = dict(base)
            set_item(feat, it, args.team)
            if args.cost_adjust and cost > 0:
                apply_cost_adjust(feat, args.team, cost)

            X = dict_to_X(feat, feature_cols)
            p_win, p_dg, p_dx = predict_all(win_b, dg_b, dx_b, X)
            tv, tdg, tdx = team_view(args.team, p_win, p_dg, p_dx)
            results.append(("ITEM", f"BUY({it})", tv - tv0, tv, tdg, tdx))

    # ---- wards (no location yet; “do it now”)
    if args.mode in ("wards","both"):
        results.append(("WARD", "HOLD()", 0.0, tv0, tdg0, tdx0))
        for kind in ("obs","sen"):
            feat = dict(base)
            add_ward(feat, args.team, kind)
            X = dict_to_X(feat, feature_cols)
            p_win, p_dg, p_dx = predict_all(win_b, dg_b, dx_b, X)
            tv, tdg, tdx = team_view(args.team, p_win, p_dg, p_dx)
            results.append(("WARD", f"PLACE_{kind.upper()}()", tv - tv0, tv, tdg, tdx))

    results.sort(key=lambda x: (x[3], x[2]), reverse=True)

    print(f"\nTop-{args.topk} actions (ranked by team_win_prob):")
    for i, (typ, act, dwin, tv, tdg, tdx) in enumerate(results[:args.topk], 1):
        print(f"{i:02d}. [{typ}] {act:20s}  Δwin={dwin:+.4f}  win={tv:.4f}  dG={tdg:+.0f}  dX={tdx:+.0f}")

if __name__ == "__main__":
    main()
