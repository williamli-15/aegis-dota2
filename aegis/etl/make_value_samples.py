import argparse
import os
import polars as pl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches_glob", default="data/processed/matches/*.parquet")
    ap.add_argument("--team_glob", default="data/processed/state_team_minute/*.parquet")
    ap.add_argument("--player_glob", default="data/processed/state_player_minute/*.parquet")
    ap.add_argument("--out", default="data/processed/value_samples_h180.parquet")
    ap.add_argument("--horizon_sec", type=int, default=180)  # 3 minutes
    ap.add_argument("--compression", default="zstd")
    args = ap.parse_args()

    H = args.horizon_sec
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    matches = (
        pl.scan_parquet(args.matches_glob)
        .select(["match_id", "radiant_win"])
    )

    team = (
        pl.scan_parquet(args.team_glob)
        .select(["match_id", "patch", "t", "radiant_gold_adv", "radiant_xp_adv"])
    )

    player = (
        pl.scan_parquet(args.player_glob)
        .select(["match_id", "patch", "t", "team", "gold", "xp", "lh", "dn"])
    )

    # --- aggregate per team per minute
    agg = (
        player.group_by(["match_id", "patch", "t", "team"])
        .agg([
            pl.sum("gold").alias("gold_sum"),
            pl.sum("xp").alias("xp_sum"),
            pl.sum("lh").alias("lh_sum"),
            pl.sum("dn").alias("dn_sum"),
            pl.mean("gold").alias("gold_mean"),
            pl.mean("xp").alias("xp_mean"),
            pl.max("gold").alias("gold_max"),
            pl.max("xp").alias("xp_max"),
        ])
    )

    rad = (
        agg.filter(pl.col("team") == 0)
        .drop("team")
        .rename({
            "gold_sum": "rad_gold_sum",
            "xp_sum": "rad_xp_sum",
            "lh_sum": "rad_lh_sum",
            "dn_sum": "rad_dn_sum",
            "gold_mean": "rad_gold_mean",
            "xp_mean": "rad_xp_mean",
            "gold_max": "rad_gold_max",
            "xp_max": "rad_xp_max",
        })
    )

    dire = (
        agg.filter(pl.col("team") == 1)
        .drop("team")
        .rename({
            "gold_sum": "dire_gold_sum",
            "xp_sum": "dire_xp_sum",
            "lh_sum": "dire_lh_sum",
            "dn_sum": "dire_dn_sum",
            "gold_mean": "dire_gold_mean",
            "xp_mean": "dire_xp_mean",
            "gold_max": "dire_gold_max",
            "xp_max": "dire_xp_max",
        })
    )

    feat = rad.join(dire, on=["match_id", "patch", "t"], how="inner")

    # diffs (radiant - dire)
    feat = feat.with_columns([
        (pl.col("rad_gold_sum") - pl.col("dire_gold_sum")).alias("gold_sum_diff"),
        (pl.col("rad_xp_sum") - pl.col("dire_xp_sum")).alias("xp_sum_diff"),
        (pl.col("rad_lh_sum") - pl.col("dire_lh_sum")).alias("lh_sum_diff"),
        (pl.col("rad_dn_sum") - pl.col("dire_dn_sum")).alias("dn_sum_diff"),
        (pl.col("rad_gold_mean") - pl.col("dire_gold_mean")).alias("gold_mean_diff"),
        (pl.col("rad_xp_mean") - pl.col("dire_xp_mean")).alias("xp_mean_diff"),
        (pl.col("rad_gold_max") - pl.col("dire_gold_max")).alias("gold_max_diff"),
        (pl.col("rad_xp_max") - pl.col("dire_xp_max")).alias("xp_max_diff"),
        (pl.col("t") / 60.0).alias("t_min"),
    ])

    # join team advantages (current)
    feat = feat.join(team, on=["match_id", "patch", "t"], how="inner")

    # create future advantages table aligned to current t
    future = (
        team.select(["match_id", "patch", "t", "radiant_gold_adv", "radiant_xp_adv"])
        .with_columns((pl.col("t") - H).alias("t"))  # shift back so future aligns to current
        .rename({
            "radiant_gold_adv": "radiant_gold_adv_f",
            "radiant_xp_adv": "radiant_xp_adv_f",
        })
    )

    df = feat.join(future, on=["match_id", "patch", "t"], how="left")

    # keep rows where future exists and current adv exists
    df = df.drop_nulls(["radiant_gold_adv", "radiant_xp_adv", "radiant_gold_adv_f", "radiant_xp_adv_f"])

    # labels
    df = df.with_columns([
        (pl.col("radiant_gold_adv_f") - pl.col("radiant_gold_adv")).alias("y_dgold_adv"),
        (pl.col("radiant_xp_adv_f") - pl.col("radiant_xp_adv")).alias("y_dxp_adv"),
    ]).drop(["radiant_gold_adv_f", "radiant_xp_adv_f"])

    # join win label
    df = df.join(matches, on="match_id", how="left").drop_nulls(["radiant_win"])
    df = df.with_columns(pl.col("radiant_win").cast(pl.Int8).alias("y_win")).drop("radiant_win")

    # NOTE: do NOT include patch as a model feature by default; keep it for slicing
    # Write parquet (streaming if available)
    try:
        df.sink_parquet(args.out, compression=args.compression)
    except Exception:
        df.collect(streaming=True).write_parquet(args.out, compression=args.compression)

    print(f"Wrote: {args.out} (horizon={H}s)")

if __name__ == "__main__":
    main()
