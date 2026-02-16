"""
python -m aegis.etl.make_team_wards_minute \
  --out data/processed/team_wards_minute.parquet
"""

import argparse, os
import polars as pl

WINDOW_MIN = 5  # rolling window length in minutes

WARD_EVENTS = [
    "ward_obs_place",
    "ward_obs_left",
    "ward_sen_place",
    "ward_sen_left",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events_glob", default="data/processed/events/*.parquet")
    ap.add_argument("--team_glob", default="data/processed/state_team_minute/*.parquet")
    ap.add_argument("--out", default="data/processed/team_wards_minute.parquet")
    ap.add_argument("--compression", default="zstd")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # minute grid (match,patch,t) x team
    base_mt = (
        pl.scan_parquet(args.team_glob)
        .select(["match_id","patch","t"])
        .unique()
    )
    teams = pl.DataFrame({"team":[0,1]}).lazy()
    base = base_mt.join(teams, how="cross").collect(streaming=True)

    # ward events -> minute bucket (clip t<0 to 0) -> counts per minute
    ev = (
        pl.scan_parquet(args.events_glob)
        .filter(pl.col("event_type").is_in(WARD_EVENTS))
        .select(["match_id","patch","t","team","event_type"])
        .with_columns((pl.col("t") // 60 * 60).alias("tb"))
        .with_columns(pl.when(pl.col("tb") < 0).then(0).otherwise(pl.col("tb")).alias("t"))
        .drop("tb")
        .group_by(["match_id","patch","t","team","event_type"])
        .agg(pl.len().alias("cnt"))
        .collect(streaming=True)
    )

    wide = ev.pivot(
        values="cnt",
        index=["match_id","patch","t","team"],
        columns="event_type",
        aggregate_function="sum",
    )

    df = base.join(wide, on=["match_id","patch","t","team"], how="left")

    # ensure all ward cols exist + fill nulls
    for c in WARD_EVENTS:
        if c not in df.columns:
            df = df.with_columns(pl.lit(0).alias(c))
        df = df.with_columns(pl.col(c).fill_null(0).cast(pl.Int16))

    df = df.sort(["match_id","patch","team","t"])

    # cumulative counts
    df = df.with_columns([
        pl.col("ward_obs_place").cum_sum().over(["match_id","patch","team"]).alias("cum_obs_place"),
        pl.col("ward_obs_left").cum_sum().over(["match_id","patch","team"]).alias("cum_obs_left"),
        pl.col("ward_sen_place").cum_sum().over(["match_id","patch","team"]).alias("cum_sen_place"),
        pl.col("ward_sen_left").cum_sum().over(["match_id","patch","team"]).alias("cum_sen_left"),
    ])

    # active wards (clamp >=0 just in case of rare log inconsistencies)
    active_obs_raw = (pl.col("cum_obs_place") - pl.col("cum_obs_left"))
    active_sen_raw = (pl.col("cum_sen_place") - pl.col("cum_sen_left"))

    df = df.with_columns([
        pl.when(active_obs_raw < 0).then(0).otherwise(active_obs_raw).alias("active_obs"),
        pl.when(active_sen_raw < 0).then(0).otherwise(active_sen_raw).alias("active_sen"),
    ])

    # rolling 5min counts via cum_sum diffs (window = 5 rows == 5 minutes)
    df = df.with_columns([
        (pl.col("cum_obs_place") - pl.col("cum_obs_place").shift(WINDOW_MIN).over(["match_id","patch","team"]).fill_null(0)).alias("obs_place_5m"),
        (pl.col("cum_obs_left")  - pl.col("cum_obs_left").shift(WINDOW_MIN).over(["match_id","patch","team"]).fill_null(0)).alias("obs_left_5m"),
        (pl.col("cum_sen_place") - pl.col("cum_sen_place").shift(WINDOW_MIN).over(["match_id","patch","team"]).fill_null(0)).alias("sen_place_5m"),
        (pl.col("cum_sen_left")  - pl.col("cum_sen_left").shift(WINDOW_MIN).over(["match_id","patch","team"]).fill_null(0)).alias("sen_left_5m"),
    ])

    # rename 1m counts to explicit names
    df = df.with_columns([
        pl.col("ward_obs_place").alias("obs_place_1m"),
        pl.col("ward_obs_left").alias("obs_left_1m"),
        pl.col("ward_sen_place").alias("sen_place_1m"),
        pl.col("ward_sen_left").alias("sen_left_1m"),
    ])

    keep = [
        "match_id","patch","t","team",
        "active_obs","active_sen",
        "obs_place_1m","obs_left_1m","sen_place_1m","sen_left_1m",
        "obs_place_5m","obs_left_5m","sen_place_5m","sen_left_5m",
    ]
    df = df.select(keep)

    df.write_parquet(args.out, compression=args.compression)
    print("Wrote:", args.out, "rows=", df.height, "cols=", len(df.columns))

if __name__ == "__main__":
    main()
