import argparse, os
import polars as pl

WINDOW_MIN = 5  # rolling window length in minutes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events_glob", default="data/processed/events/*.parquet")
    ap.add_argument("--teamfights_glob", default="data/processed/teamfights/*.parquet")
    ap.add_argument("--team_glob", default="data/processed/state_team_minute/*.parquet")
    ap.add_argument("--out", default="data/processed/team_skirmish_minute.parquet")
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

    # -----------------------
    # (A) Kills per team per minute
    # -----------------------
    kills = (
        pl.scan_parquet(args.events_glob)
        .filter(pl.col("event_type") == "kill")
        .filter(pl.col("t") >= 0)
        .select(["match_id","patch","t","team"])
        .with_columns((pl.col("t") // 60 * 60).alias("t"))
        .group_by(["match_id","patch","t","team"])
        .agg(pl.len().alias("kills_1m"))
        .collect(streaming=True)
    )

    kd = base.join(kills, on=["match_id","patch","t","team"], how="left")
    kd = kd.with_columns(pl.col("kills_1m").fill_null(0).cast(pl.Int16))

    # deaths_1m = opponent kills_1m
    opp = (
        kd.select(["match_id","patch","t","team","kills_1m"])
        .with_columns((1 - pl.col("team")).alias("team"))
        .rename({"kills_1m":"deaths_1m"})
    )
    kd = kd.join(opp.select(["match_id","patch","t","team","deaths_1m"]),
                 on=["match_id","patch","t","team"], how="left")
    kd = kd.with_columns(pl.col("deaths_1m").fill_null(0).cast(pl.Int16))

    kd = kd.sort(["match_id","patch","team","t"])

    # rolling 5min via cum_sum diff (window = 5 rows because minute grid)
    kd = kd.with_columns([
        pl.col("kills_1m").cum_sum().over(["match_id","patch","team"]).alias("cum_kills"),
        pl.col("deaths_1m").cum_sum().over(["match_id","patch","team"]).alias("cum_deaths"),
    ])

    kd = kd.with_columns([
        (pl.col("cum_kills") - pl.col("cum_kills").shift(WINDOW_MIN).over(["match_id","patch","team"]).fill_null(0)).alias("kills_5m"),
        (pl.col("cum_deaths") - pl.col("cum_deaths").shift(WINDOW_MIN).over(["match_id","patch","team"]).fill_null(0)).alias("deaths_5m"),
    ]).with_columns([
        (pl.col("kills_5m") - pl.col("deaths_5m")).alias("netkills_5m"),
    ]).drop(["cum_kills","cum_deaths"])

    # -----------------------
    # (B) Teamfight context per match per minute (global, then broadcast to team rows)
    # -----------------------
    fights = (
        pl.scan_parquet(args.teamfights_glob)
        .select(["match_id","patch","fight_id","start","deaths"])
        .with_columns((pl.col("start") // 60 * 60).alias("t"))
        .group_by(["match_id","patch","t"])
        .agg([
            pl.len().alias("fights_1m"),
            pl.sum("deaths").alias("fight_deaths_1m"),
        ])
        .collect(streaming=True)
    )

    mgrid = base_mt.collect(streaming=True).join(fights, on=["match_id","patch","t"], how="left")
    mgrid = mgrid.with_columns([
        pl.col("fights_1m").fill_null(0).cast(pl.Int16),
        pl.col("fight_deaths_1m").fill_null(0).cast(pl.Int16),
    ]).sort(["match_id","patch","t"])

    mgrid = mgrid.with_columns([
        pl.col("fights_1m").cum_sum().over(["match_id","patch"]).alias("cum_fights"),
        pl.col("fight_deaths_1m").cum_sum().over(["match_id","patch"]).alias("cum_fight_deaths"),
    ]).with_columns([
        (pl.col("cum_fights") - pl.col("cum_fights").shift(WINDOW_MIN).over(["match_id","patch"]).fill_null(0)).alias("fights_5m"),
        (pl.col("cum_fight_deaths") - pl.col("cum_fight_deaths").shift(WINDOW_MIN).over(["match_id","patch"]).fill_null(0)).alias("fight_deaths_5m"),
    ]).drop(["cum_fights","cum_fight_deaths"])

    # time since last fight minute (based on fights_1m>0)
    mgrid = mgrid.with_columns([
        pl.when(pl.col("fights_1m") > 0).then(pl.col("t")).otherwise(None).alias("fight_t"),
    ]).with_columns([
        pl.col("fight_t").fill_null(strategy="forward").over(["match_id","patch"]).alias("last_fight_t")
    ]).with_columns([
        pl.when(pl.col("last_fight_t").is_null()).then(-1)
          .otherwise(pl.col("t") - pl.col("last_fight_t"))
          .cast(pl.Int32)
          .alias("time_since_fight")
    ]).drop(["fight_t","last_fight_t"])

    # broadcast match-level fight features to each team row
    kd = kd.join(mgrid.select(["match_id","patch","t","fights_1m","fight_deaths_1m","fights_5m","fight_deaths_5m","time_since_fight"]),
                 on=["match_id","patch","t"], how="left")
    kd = kd.with_columns([
        pl.col("fights_1m").fill_null(0).cast(pl.Int16),
        pl.col("fight_deaths_1m").fill_null(0).cast(pl.Int16),
        pl.col("fights_5m").fill_null(0).cast(pl.Int16),
        pl.col("fight_deaths_5m").fill_null(0).cast(pl.Int16),
        pl.col("time_since_fight").fill_null(-1).cast(pl.Int32),
    ])

    kd.write_parquet(args.out, compression=args.compression)
    print("Wrote:", args.out, "rows=", kd.height, "cols=", len(kd.columns))

if __name__ == "__main__":
    main()
