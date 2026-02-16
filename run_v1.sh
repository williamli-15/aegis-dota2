#!/usr/bin/env bash
set -euo pipefail

# ===== Config =====
HORIZON_SEC="${HORIZON_SEC:-180}"
MIN_MATCHES="${MIN_MATCHES:-800}"
MIN_T50="${MIN_T50:-600}"            # seconds
MAX_PRESENCE="${MAX_PRESENCE:-0.90}" # <= 90% of team-games
MAX_VOCAB="${MAX_VOCAB:-80}"

RAW_VALUE="data/processed/value_samples_h${HORIZON_SEC}.parquet"
VOCAB="data/processed/item_vocab_v1.txt"
TEAM_ITEMS="data/processed/team_items_minute.parquet"
TEAM_WARDS="data/processed/team_wards_minute.parquet"
TEAM_SKIRMISH="data/processed/team_skirmish_minute.parquet"
VALUE_V1="data/processed/value_samples_v1_h${HORIZON_SEC}.parquet"

ART_DIR="artifacts/value_xgb_v1_h${HORIZON_SEC}"

echo "=== Aegis v1 pipeline ==="
echo "HORIZON_SEC=$HORIZON_SEC"
echo "VOCAB: min_matches=$MIN_MATCHES min_t50=$MIN_T50 max_presence=$MAX_PRESENCE max_vocab=$MAX_VOCAB"
echo

# ===== Sanity checks =====
if [[ ! -f "$RAW_VALUE" ]]; then
  echo "[ERROR] Missing base value samples: $RAW_VALUE"
  echo "Run first: python -m aegis.etl.make_value_samples --out $RAW_VALUE --horizon_sec $HORIZON_SEC"
  exit 1
fi

mkdir -p data/processed artifacts

# ===== Step 1: Item vocab =====
echo ">> (1/6) make_item_vocab -> $VOCAB"
python -m aegis.etl.make_item_vocab \
  --out "$VOCAB" \
  --min_matches "$MIN_MATCHES" \
  --min_t50 "$MIN_T50" \
  --max_presence "$MAX_PRESENCE" \
  --max_vocab "$MAX_VOCAB"

echo

# ===== Step 2: Team items per minute =====
echo ">> (2/6) make_team_items_minute -> $TEAM_ITEMS"
python -m aegis.etl.make_team_items_minute \
  --item_vocab "$VOCAB" \
  --out "$TEAM_ITEMS"

echo

# ===== Step 3: Team wards per minute (active + 1m/5m) =====
echo ">> (3/6) make_team_wards_minute -> $TEAM_WARDS"
python -m aegis.etl.make_team_wards_minute \
  --out "$TEAM_WARDS"

# quick sanity: active wards should never be negative
echo "   sanity: min(active_obs), min(active_sen)"
python - <<'PY'
import duckdb
con=duckdb.connect()
print(con.execute("""
select min(active_obs) as min_active_obs, min(active_sen) as min_active_sen
from read_parquet('data/processed/team_wards_minute.parquet')
""").fetchall())
PY

echo

# ===== Step 4: Team skirmish per minute (kills + fights + time_since_fight) =====
echo ">> (4/6) make_team_skirmish_minute -> $TEAM_SKIRMISH"
python -m aegis.etl.make_team_skirmish_minute \
  --out "$TEAM_SKIRMISH"

echo

# ===== Step 5: Build value_samples_v1 =====
echo ">> (5/6) make_value_samples_v1 -> $VALUE_V1"
python -m aegis.etl.make_value_samples_v1 \
  --base "$RAW_VALUE" \
  --items "$TEAM_ITEMS" \
  --wards "$TEAM_WARDS" \
  --skirmish "$TEAM_SKIRMISH" \
  --out "$VALUE_V1"

echo

# ===== Step 6: Train value v1 =====
echo ">> (6/6) train_value_xgb -> $ART_DIR"
python -m aegis.models.value.train_value_xgb \
  --value "$VALUE_V1" \
  --splits_dir data/splits \
  --out_dir "$ART_DIR"

echo
echo "=== DONE ==="
echo "Artifacts in: $ART_DIR"
