import argparse, os
import polars as pl

def prefix_cols(df: pl.LazyFrame, prefix: str, keep=("match_id","patch","t")):
    ren = {}
    for c in df.columns:
        if c in keep:
            continue
        ren[c] = prefix + c
    return df.rename(ren)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/processed/value_samples_h180.parquet")
    ap.add_argument("--items", default="data/processed/team_items_minute.parquet")
    ap.add_argument("--wards", default="data/processed/team_wards_minute.parquet")
    ap.add_argument("--skirmish", default="data/processed/team_skirmish_minute.parquet")
    ap.add_argument("--out", default="data/processed/value_samples_v1_h180.parquet")
    ap.add_argument("--compression", default="zstd")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    base = pl.scan_parquet(args.base)
    items = pl.scan_parquet(args.items)
    wards = pl.scan_parquet(args.wards)
    sk = pl.scan_parquet(args.skirmish)

    rad_items = prefix_cols(items.filter(pl.col("team")==0).drop("team"), "rad_")
    dire_items = prefix_cols(items.filter(pl.col("team")==1).drop("team"), "dire_")

    rad_wards = prefix_cols(wards.filter(pl.col("team")==0).drop("team"), "rad_")
    dire_wards = prefix_cols(wards.filter(pl.col("team")==1).drop("team"), "dire_")

    rad_sk = prefix_cols(sk.filter(pl.col("team")==0).drop("team"), "rad_")
    dire_sk = prefix_cols(sk.filter(pl.col("team")==1).drop("team"), "dire_")

    df = (base
          .join(rad_items, on=["match_id","patch","t"], how="left")
          .join(dire_items, on=["match_id","patch","t"], how="left")
          .join(rad_wards, on=["match_id","patch","t"], how="left")
          .join(dire_wards, on=["match_id","patch","t"], how="left")
          .join(rad_sk, on=["match_id","patch","t"], how="left")
          .join(dire_sk, on=["match_id","patch","t"], how="left")
    )

    # fill nulls -> 0 / -1
    new_cols = [c for c in df.columns if c.startswith("rad_") or c.startswith("dire_")]
    # time_since_fight should keep -1; everything else fill 0
    exprs = []
    for c in new_cols:
        if c.endswith("time_since_fight"):
            exprs.append(pl.col(c).fill_null(-1).alias(c))
        else:
            exprs.append(pl.col(c).fill_null(0).alias(c))
    df = df.with_columns(exprs)

    # diffs (rad - dire)
    diff_exprs = []
    for c in new_cols:
        if c.startswith("rad_"):
            d = "dire_" + c[len("rad_"):]
            if d in new_cols:
                diff_exprs.append((pl.col(c) - pl.col(d)).alias("diff_" + c[len("rad_"):]))
    df = df.with_columns(diff_exprs)

    df.sink_parquet(args.out, compression=args.compression)
    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
