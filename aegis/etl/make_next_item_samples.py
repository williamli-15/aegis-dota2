import argparse, os
import polars as pl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value_v1", default="data/processed/value_samples_v1_h180.parquet")
    ap.add_argument("--events_glob", default="data/processed/events/*.parquet")
    ap.add_argument("--item_vocab", default="data/processed/item_vocab_v1.txt")
    ap.add_argument("--out", default="data/processed/next_item_samples.parquet")
    ap.add_argument("--lead_sec", type=int, default=60, help="decision time = floor(first_t/60)*60 - lead")
    ap.add_argument("--min_first_t", type=int, default=0)
    ap.add_argument("--max_rows", type=int, default=0, help="0=no cap (random sample)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compression", default="zstd")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    vocab = [x.strip() for x in open(args.item_vocab) if x.strip()]
    if not vocab:
        raise RuntimeError("Empty item_vocab")

    # base features from value_v1 (drop y_* labels)
    base = pl.scan_parquet(args.value_v1)
    cols = base.collect_schema().names()
    base_cols = [c for c in cols if not c.startswith("y_")]
    base = base.select(base_cols)

    # first-buy per match/team/item (restrict vocab)
    fb = (
        pl.scan_parquet(args.events_glob)
        .filter(pl.col("event_type") == "purchase")
        .filter(pl.col("t") >= args.min_first_t)
        .filter(pl.col("key").is_in(vocab))
        .select(["match_id","patch","team","key","t"])
        .group_by(["match_id","patch","team","key"])
        .agg(pl.min("t").alias("first_t"))
    )

    # decision time t (minute aligned) and clamp to 0
    fb = fb.with_columns([
        ((pl.col("first_t") // 60) * 60).alias("t_bucket"),
    ]).with_columns([
        (pl.col("t_bucket") - args.lead_sec).alias("t")
    ]).with_columns([
        pl.when(pl.col("t") < 0).then(0).otherwise(pl.col("t")).alias("t")
    ]).drop("t_bucket")

    # label
    fb = fb.rename({"key": "y_next_item"})

    # join features at (match_id, patch, t)
    ds = (
        fb.join(base, on=["match_id","patch","t"], how="inner")
        .with_columns([
            (pl.col("first_t") - pl.col("t")).alias("dt_to_buy"),
        ])
    )

    # optional cap for speed
    if args.max_rows and args.max_rows > 0:
        ds = ds.collect().sample(n=min(args.max_rows, ds.collect().height), seed=args.seed).lazy()

    # collect + write
    ds = ds.collect()
    ds.write_parquet(args.out, compression=args.compression)

    print("Wrote:", args.out, "rows=", ds.height, "cols=", len(ds.columns))
    print("label uniq:", ds["y_next_item"].n_unique())

if __name__ == "__main__":
    main()
