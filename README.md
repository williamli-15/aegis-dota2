## aegis-dota2

<img width="2194" height="1258" alt="image" src="https://github.com/user-attachments/assets/572a5502-1b4b-4e5e-89cd-12edabc88ab8" />

A structured, ProAct-style offline decision system for **Dota 2** built from **OpenDota parsed match logs**.
It turns match timelines into a **value model** (win + Δadv) and **policy proposals** (next item), then runs **generate-and-score** to rank feasible candidate actions.

### What’s inside (Phase 0–1)

* **ETL (JSONL → Parquet)**: matches, minute-level states, event stream, teamfights
* **Value Model (XGBoost)**
  Predicts `P(win)` + `Δgold_adv@3m` + `Δxp_adv@3m` (patch-held-out evaluation)
* **Next-item Policy (XGBoost, 80 classes)**
  Proposal model for candidate next items
* **Generate-and-Score evaluation**
  Propose Top-K items → apply feasibility/time-band filters → rank by value model

### Key results (TEST patch holdout)

From `artifacts/phase1_report.md`:

* **Value v1 > v0** (added items/wards/skirmish features)
* **Next-item policy**: Top-20 recall ≈ **0.889**
* **Generate-and-score (feasible/time-band)**: positive proxy uplift vs baseline and improved hit@1 on feasible subset

### Data

* Source: OpenDota `/parsedMatches` + `/matches/{match_id}`
* Current patches: **7.38 / 7.39 / 7.40** (patch ids 57/58/59)

### Quickstart

```bash
# 0) Create splits (patch-based)
python -m aegis.etl.make_splits_from_processed --test_patch 59 --val_patch 58 --val_frac 0.1

# 1) Build v1 pipeline (vocab → team features → value samples → value model)
./run_v1.sh

# 2) Train next-item policy (from next_item_samples.parquet)
python -m aegis.models.policy.train_next_item_policy \
  --data data/processed/next_item_samples.parquet \
  --out_dir artifacts/next_item_policy_xgb \
  --max_train_rows 300000 --topk 20

# 3) Generate-and-score eval (feasible/time-band)
python -m aegis.mvp.run_generate_and_score_eval_v2 \
  --n_eval 5000 --topk 20 \
  --feasible --fetch_constants \
  --min_item_cost 2000 --time_band_sec 900 --gold_buffer 800
```

### Repo structure (high level)

* `aegis/etl/`: JSONL → Parquet, vocab/features, sample builders
* `aegis/models/value/`: value model training
* `aegis/models/policy/`: next-item policy training
* `aegis/mvp/`: scoring + generate-and-score eval

### Next steps (Phase 2)

* Ward-where policy: `PLACE_OBS(x,y)` / `PLACE_SEN(x,y)` with survival supervision from `*_left_log`
* Fight participation policy + plan tokens for multi-step planning
