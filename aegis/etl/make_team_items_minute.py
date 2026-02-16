import argparse, os
import polars as pl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events_glob", default="data/processed/events/*.parquet")
    ap.add_argument("--team_glob", default="data/processed/state_team_minute/*.parquet")
    ap.add_argument("--item_vocab", default="data/processed/item_vocab_v1.txt")
    ap.add_argument("--out", default="data/processed/team_items_minute.parquet")
    ap.add_argument("--compression", default="zstd")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    vocab = [x.strip() for x in open(args.item_vocab) if x.strip()]
    if not vocab:
        raise RuntimeError("Empty vocab file")

    # minute grid (match,patch,t) x team
    base = (
        pl.scan_parquet(args.team_glob)
        .select(["match_id","patch","t"])
        .unique()
    )
    teams = pl.DataFrame({"team":[0,1]}).lazy()
    base = base.join(teams, how="cross")

    # purchase events -> minute bucket (clip t<0 to 0) -> count
    ev = (
        pl.scan_parquet(args.events_glob)
        .filter(pl.col("event_type") == "purchase")
        .filter(pl.col("key").is_in(vocab))
        .select(["match_id","patch","t","team","key"])
        .with_columns((pl.col("t") // 60 * 60).alias("tb"))
        .with_columns(
            pl.when(pl.col("tb") < 0).then(0).otherwise(pl.col("tb")).alias("t")
        )
        .drop("tb")
        .group_by(["match_id","patch","t","team","key"])
        .agg(pl.len().alias("cnt"))
        .collect(streaming=True)
    )

    # pivot to wide
    wide = ev.pivot(
        values="cnt",
        index=["match_id","patch","t","team"],
        columns="key",
        aggregate_function="sum",
    )

    # join full grid + fill missing with 0
    df = base.collect(streaming=True).join(wide, on=["match_id","patch","t","team"], how="left")

    # ensure all vocab columns exist
    for k in vocab:
        if k not in df.columns:
            df = df.with_columns(pl.lit(0).alias(k))
    df = df.with_columns([pl.col(k).fill_null(0).cast(pl.Int16) for k in vocab])

    # cumulative -> binary has_
    df = df.sort(["match_id","patch","team","t"])
    for k in vocab:
        df = df.with_columns(
            (pl.col(k).cum_sum().over(["match_id","patch","team"]) > 0)
            .cast(pl.Int8)
            .alias(f"has_{k}")
        )
    # keep
    keep = ["match_id","patch","t","team"] + [f"has_{k}" for k in vocab]
    df = df.select(keep)

    df.write_parquet(args.out, compression=args.compression)
    print("Wrote:", args.out, "rows=", df.height, "cols=", len(df.columns))

if __name__ == "__main__":
    main()
