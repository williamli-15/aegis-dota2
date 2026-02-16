1) ✅ Value v1 真的更强了（而且提升很大）

你 v0（econ-only）是：

AUC ~ 0.768

dGold RMSE ~ 2761

dXP RMSE ~ 5034

v1（econ+items+wards+skirmish）变成：

AUC 0.7807

dGold RMSE 2279（大幅降）

dXP RMSE 4271（大幅降）

这说明：items/wards/skirmish 这些“可控变量”确实被模型学进去了，可以当打分器。



结果非常漂亮：**v1 比 v0 明显更强**（AUC 0.7807↑、Δadv 的 RMSE 大幅下降），warnings 都是 polars 的 deprecation/perf 提示，**可以无视**。下一步可以进 **Generate-and-Score MVP** 了。

下面给你一个“能立刻用起来”的下一步：**写一个 counterfactual scorer** ——给定某个 match 的某一分钟状态，枚举候选 **买某个 item / 插一根 obs/sen（不含坐标）**，用你训练好的 `value_xgb_v1_h180` 打分并输出 top-K。

> 这个 MVP 的目标是：先把 *“候选→value打分→选优”* 的闭环跑通；后面再接 **候选生成 policy**（next-item/ward-where）以及更强的 world model。

---

# 1) 新增模块：`aegis/mvp/score_actions.py`

先建目录：

```bash
mkdir -p aegis/mvp
touch aegis/mvp/__init__.py
```

保存下面脚本到：`aegis/mvp/score_actions.py`

```python
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
        if os.path.exists(args.item_constants):
            with open(args.item_constants, "r") as f:
                data = json.load(f)
            item_cost = {k: int(v.get("cost", 0)) for k, v in data.items()}
        elif args.fetch_constants:
            import requests
            url = "https://api.opendota.com/api/constants/items"
            data = requests.get(url, timeout=30).json()
            os.makedirs(os.path.dirname(args.item_constants), exist_ok=True)
            with open(args.item_constants, "w") as f:
                json.dump(data, f)
            item_cost = {k: int(v.get("cost", 0)) for k, v in data.items()}
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
```

---

# 2) 运行方式（建议你先在 test patch 上随便挑一局玩一下）

拿一局 test match：

```bash
MID=$(head -n 1 data/splits/test.txt)
python - <<PY
import polars as pl
mid = int("$MID")
df = (pl.scan_parquet("data/processed/value_samples_v1_h180.parquet")
      .filter(pl.col("match_id")==mid)
      .select(["t"]).collect())
print("t_min:", int(df["t"].min()), "t_max:", int(df["t"].max()))
print("first 15 t:", df["t"][:15].to_list())
PY

```

跑 scorer（Radiant）：

```bash
python -m aegis.mvp.score_actions \
  --match_id $MID --t 900 --team 0 --mode both --topk 15
```

跑 scorer（Dire）：

```bash
python -m aegis.mvp.score_actions \
  --match_id $MID --t 900 --team 1 --mode both --topk 15
```

如果你想过滤掉“组件类”的建议（只看大件），用 cost 过滤（第一次会下载并缓存 OpenDota constants）：

```bash
python -m aegis.mvp.score_actions \
  --match_id $MID --t 900 --team 0 --mode items --topk 15 \
  --min_item_cost 2000 --fetch_constants
```

---

# 3) 你现在问的几件事——结论

### ✅ v1 输出没问题，提升很明显

* AUC 0.7807 vs v0 0.7681
* Δadv RMSE 大幅下降（dGold 2761→2280，dXP 5034→4271）
  这说明 items/wards/skirmish 特征确实让 value 更“行动敏感”。

### ⚠️ warnings 不用管

全是 polars 的 streaming/pivot 参数 deprecation + LazyFrame schema warning，不影响结果。

### “Phase 1 够不够用？要不要 per-fight per-player 表？”

**现在够用。**

* Value v1 + Generate-and-Score MVP 不需要 per-fight per-player
* 等你们做 **fight participation policy / 更强 world model** 再扩展 `teamfight_players`（Phase 2）

---

# 下一步之后是什么（你跑完 scorer 我就接着给）

你跑上面的 `score_actions.py` 看看输出是否“像那么回事”。
下一步我会给你一个 **offline eval harness**（hit@k + estimated uplift 曲线），让你们能写进 paper/PRD 的 evaluation section，然后再接 **候选生成 policy（next-item / ward-where）**。
