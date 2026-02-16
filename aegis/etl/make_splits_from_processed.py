import argparse, os
import polars as pl

FEW_SHOT_NS = [25, 50, 100, 200, 500, 1000]

def write(path, mids):
    with open(path, "w") as f:
        for m in mids:
            f.write(str(int(m)) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches_glob", default="data/processed/matches/*.parquet")
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--test_patch", type=int, required=True) # 59
    ap.add_argument("--val_patch", type=int, required=True)  # 58
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pl.scan_parquet(args.matches_glob).select(
        ["match_id","patch","start_time"]
    ).collect()

    test_df = df.filter(pl.col("patch") == args.test_patch).sort("start_time")
    val_df  = df.filter(pl.col("patch") == args.val_patch).sort("start_time")
    other_df = df.filter((pl.col("patch") != args.test_patch) & (pl.col("patch") != args.val_patch))

    test = test_df["match_id"].to_list()
    k = int((1.0 - args.val_frac) * len(val_df))
    train = other_df["match_id"].to_list() + val_df["match_id"][:k].to_list()
    val = val_df["match_id"][k:].to_list()

    write(os.path.join(args.out_dir, "train.txt"), train)
    write(os.path.join(args.out_dir, "val.txt"), val)
    write(os.path.join(args.out_dir, "test.txt"), test)

    for n in FEW_SHOT_NS:
        n_eff = min(n, len(test))
        write(os.path.join(args.out_dir, f"test_calib_{n}.txt"), test[:n_eff])
        write(os.path.join(args.out_dir, f"test_eval_{n}.txt"), test[n_eff:])

    print("Done.")
    print("train", len(train), "val", len(val), "test", len(test))

if __name__ == "__main__":
    main()
