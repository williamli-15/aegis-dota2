import argparse, os, json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

ALWAYS_INCLUDE = {
    "blink", "black_king_bar", "force_staff", "glimmer_cape",
    "ultimate_scepter", "aghanims_shard",
}

def load_features(path: str) -> List[str]:
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]

def load_vocab(path: str) -> List[str]:
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]

def load_row(value_parquet: str, match_id: int, t: int) -> Dict[str, float]:
    # exact match first
    df = (
        pl.scan_parquet(value_parquet)
        .filter((pl.col("match_id") == match_id) & (pl.col("t") == t))
        .collect()
    )
    if df.height == 0:
        # pick closest available t for this match
        ts = (
            pl.scan_parquet(value_parquet)
            .filter(pl.col("match_id") == match_id)
            .select(["t"])
            .collect()
        )
        if ts.height == 0:
            raise ValueError(f"No rows for match_id={match_id} in {value_parquet}")
        t_closest = int(ts["t"].to_numpy()[np.argmin(np.abs(ts["t"].to_numpy() - t))])
        df = (
            pl.scan_parquet(value_parquet)
            .filter((pl.col("match_id") == match_id) & (pl.col("t") == t_closest))
            .collect()
        )
        print(f"[warn] t={t} not found; using closest t={t_closest}")
    row = df.row(0, named=True)
    # cast to python scalars
    out = {}
    for k, v in row.items():
        if v is None:
            continue
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out

def dict_to_X(feat: Dict[str, float], feature_cols: List[str]) -> pd.DataFrame:
    data = {c: [feat.get(c, 0.0)] for c in feature_cols}
    return pd.DataFrame(data)

def load_booster(path: str) -> xgb.Booster:
    b = xgb.Booster()
    b.load_model(path)
    return b

def predict_all(win_b: xgb.Booster, dg_b: xgb.Booster, dx_b: xgb.Booster,
                X: pd.DataFrame) -> Tuple[float, float, float]:
    dmat = xgb.DMatrix(X, feature_names=list(X.columns))
    p_win = float(win_b.predict(dmat)[0])
    p_dg = float(dg_b.predict(dmat)[0])
    p_dx = float(dx_b.predict(dmat)[0])
    return p_win, p_dg, p_dx

def team_view(team: int, p_win: float, p_dg: float, p_dx: float) -> Tuple[float, float, float]:
    # labels are radiant-centric; convert to "this team" perspective
    if team == 0:
        return p_win, p_dg, p_dx
    else:
        return 1.0 - p_win, -p_dg, -p_dx

def set_item(feat: Dict[str, float], item: str, team: int):
    rad_col = f"rad_has_{item}"
    dire_col = f"dire_has_{item}"
    diff_col = f"diff_has_{item}"

    # default missing cols to 0
    feat.setdefault(rad_col, 0.0)
    feat.setdefault(dire_col, 0.0)

    if team == 0:
        feat[rad_col] = 1.0
    else:
        feat[dire_col] = 1.0

    if diff_col in feat:
        feat[diff_col] = feat.get(rad_col, 0.0) - feat.get(dire_col, 0.0)

def add_ward(feat: Dict[str, float], team: int, kind: str):
    # kind in {"obs","sen"}
    pref = "rad_" if team == 0 else "dire_"
    other = "dire_" if team == 0 else "rad_"

    if kind == "obs":
        cols = [("active_obs", 1), ("obs_place_1m", 1), ("obs_place_5m", 1)]
    else:
        cols = [("active_sen", 1), ("sen_place_1m", 1), ("sen_place_5m", 1)]

    for base, inc in cols:
        c = pref + base
        feat[c] = feat.get(c, 0.0) + inc

        # update diff if present
        diff = "diff_" + base
        if diff in feat:
            feat.setdefault(other + base, 0.0)
            feat[diff] = feat.get("rad_" + base, 0.0) - feat.get("dire_" + base, 0.0)

def already_has_item(feat: Dict[str, float], item: str, team: int) -> bool:
    c = ("rad_" if team == 0 else "dire_") + f"has_{item}"
    return feat.get(c, 0.0) >= 0.5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value_parquet", default="data/processed/value_samples_v1_h180.parquet")
    ap.add_argument("--models_dir", default="artifacts/value_xgb_v1_h180")
    ap.add_argument("--item_vocab", default="data/processed/item_vocab_v1.txt")
    ap.add_argument("--match_id", type=int, required=True)
    ap.add_argument("--t", type=int, default=900, help="seconds (minute-aligned preferred)")
    ap.add_argument("--team", type=int, default=0, choices=[0,1], help="0=Radiant 1=Dire")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--mode", default="both", choices=["items","wards","both"])
    ap.add_argument("--min_item_cost", type=int, default=0, help="optional: filter candidates by cost >= this; needs constants")
    ap.add_argument("--item_constants", default="data/processed/opendota_items_constants.json")
    ap.add_argument("--fetch_constants", action="store_true")
    args = ap.parse_args()

    features_path = os.path.join(args.models_dir, "features.txt")
    win_path = os.path.join(args.models_dir, "win_xgb.json")
    dg_path  = os.path.join(args.models_dir, "dgold_xgb.json")
    dx_path  = os.path.join(args.models_dir, "dxp_xgb.json")

    feature_cols = load_features(features_path)
    vocab = load_vocab(args.item_vocab)

    # (optional) load item constants for cost filtering
    item_cost = None
    if args.min_item_cost > 0:

        def _cost(v):
            c = v.get("cost", 0)
            return int(c) if c is not None else 0

        if os.path.exists(args.item_constants):
            with open(args.item_constants, "r") as f:
                data = json.load(f)
            item_cost = {k: _cost(v) for k, v in data.items()}

        elif args.fetch_constants:
            import requests
            url = "https://api.opendota.com/api/constants/items"
            data = requests.get(url, timeout=30).json()
            os.makedirs(os.path.dirname(args.item_constants), exist_ok=True)
            with open(args.item_constants, "w") as f:
                json.dump(data, f)
            item_cost = {k: _cost(v) for k, v in data.items()}

        else:
            print("[warn] min_item_cost set but no constants file. Run once with --fetch_constants.")


    # load base feature row
    base = load_row(args.value_parquet, args.match_id, args.t)

    # load models
    win_b = load_booster(win_path)
    dg_b  = load_booster(dg_path)
    dx_b  = load_booster(dx_path)

    X0 = dict_to_X(base, feature_cols)
    p_win0, p_dg0, p_dx0 = predict_all(win_b, dg_b, dx_b, X0)
    tv0, tdg0, tdx0 = team_view(args.team, p_win0, p_dg0, p_dx0)

    print(f"\nBase @ match={args.match_id} t={int(base.get('t', args.t))} team={args.team} (0=Rad 1=Dire)")
    print(f"  team_win_prob={tv0:.4f}   team_dGoldAdv={tdg0:.1f}   team_dXPAdv={tdx0:.1f}")

    results = []

    # ---------- Items ----------
    if args.mode in ("items","both"):
        cand = []
        for it in vocab:
            if already_has_item(base, it, args.team):
                continue
            if args.min_item_cost > 0 and item_cost is not None:
                c = item_cost.get(it, 0)
                if c < args.min_item_cost and it not in ALWAYS_INCLUDE:
                    continue
            cand.append(it)

        for it in cand:
            feat = dict(base)  # copy
            set_item(feat, it, args.team)
            X = dict_to_X(feat, feature_cols)
            p_win, p_dg, p_dx = predict_all(win_b, dg_b, dx_b, X)
            tv, tdg, tdx = team_view(args.team, p_win, p_dg, p_dx)
            results.append((
                "ITEM", f"BUY({it})",
                tv - tv0,
                tv, tdg, tdx
            ))

    # ---------- Wards (no location yet, just “place one now”) ----------
    if args.mode in ("wards","both"):
        # HOLD baseline option
        results.append(("WARD", "HOLD()", 0.0, tv0, tdg0, tdx0))

        for kind in ("obs","sen"):
            feat = dict(base)
            add_ward(feat, args.team, kind)
            X = dict_to_X(feat, feature_cols)
            p_win, p_dg, p_dx = predict_all(win_b, dg_b, dx_b, X)
            tv, tdg, tdx = team_view(args.team, p_win, p_dg, p_dx)
            results.append((
                "WARD", f"PLACE_{kind.upper()}()",
                tv - tv0,
                tv, tdg, tdx
            ))

    # sort by team_win_prob (primary) then delta
    results.sort(key=lambda x: (x[3], x[2]), reverse=True)

    print(f"\nTop-{args.topk} actions (ranked by team_win_prob):")
    for i, (typ, act, dwin, tv, tdg, tdx) in enumerate(results[:args.topk], 1):
        print(f"{i:02d}. [{typ}] {act:18s}  Δwin={dwin:+.4f}  win={tv:.4f}  dG={tdg:+.0f}  dX={tdx:+.0f}")

if __name__ == "__main__":
    main()
