"""
python -m aegis.etl.make_item_vocab \
  --out data/processed/item_vocab_v1.txt \
  --min_matches 800 \
  --min_t50 600 \
  --max_presence 0.90 \
  --max_vocab 80
"""

import argparse, os
import duckdb
import pandas as pd

STOP_EXACT = {
    "ward_sentry","ward_observer",
    "tango","tango_single","clarity","flask","enchanted_mango",
    "blood_grenade","faerie_fire","branches",
    "tpscroll","dust","smoke_of_deceit",
    "great_famango","greater_famango",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events_glob", default="data/processed/events/*.parquet")
    ap.add_argument("--matches_glob", default="data/processed/matches/*.parquet")
    ap.add_argument("--out", default="data/processed/item_vocab_v1.txt")
    ap.add_argument("--min_matches", type=int, default=800)     # team-match count
    ap.add_argument("--min_t50", type=float, default=600.0)     # seconds (>=10min)
    ap.add_argument("--max_presence", type=float, default=0.90) # <= 90% teams
    ap.add_argument("--max_vocab", type=int, default=80)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    con = duckdb.connect()

    match_cnt = con.execute(f"select count(*) from read_parquet('{args.matches_glob}')").fetchone()[0]
    team_total = 2 * match_cnt

    df = con.execute(f"""
    WITH first_buy AS (
      SELECT
        match_id,
        team,
        key,
        MIN(t) AS first_t
      FROM read_parquet('{args.events_glob}')
      WHERE event_type='purchase'
        AND t >= 0
        AND key IS NOT NULL
        AND key NOT LIKE 'recipe_%'
      GROUP BY 1,2,3
    )
    SELECT
      key,
      COUNT(*) AS matches_with_item,
      quantile_cont(first_t, 0.5) AS t50_first_t,
      AVG(first_t) AS avg_first_t
    FROM first_buy
    GROUP BY 1
    HAVING matches_with_item >= {args.min_matches}
    ORDER BY matches_with_item DESC
    """).df()

    # filters
    df["presence"] = df["matches_with_item"] / float(team_total)
    df = df[df["t50_first_t"] >= args.min_t50].copy()
    df = df[df["presence"] <= args.max_presence].copy()

    # stoplists
    df = df[~df["key"].isin(STOP_EXACT)].copy()
    df = df[~df["key"].str.contains("mango", case=False, na=False)].copy()

    df = df.sort_values(["matches_with_item","t50_first_t"], ascending=[False, True]).head(args.max_vocab)

    with open(args.out, "w") as f:
        for k in df["key"].tolist():
            f.write(k + "\n")

    print("match_cnt:", match_cnt, "team_total:", team_total)
    print("Wrote vocab:", args.out, "size=", len(df))
    print(df.head(40).to_string(index=False))

if __name__ == "__main__":
    main()
