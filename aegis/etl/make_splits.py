import argparse, orjson, os
from collections import defaultdict

FEW_SHOT_NS = [25, 50, 100, 200, 500, 1000]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_patch", type=int, required=True)  # e.g. 59
    ap.add_argument("--val_patch", type=int, required=True)   # e.g. 58
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    by_patch = defaultdict(list)  # patch -> [(start_time, match_id)]
    with open(args.input, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            obj = orjson.loads(line)
            mid = int(obj["match_id"])
            patch = int(obj["patch"])
            st = int(obj.get("start_time", 0))
            by_patch[patch].append((st, mid))

    # sort each patch by time
    for p in by_patch:
        by_patch[p].sort(key=lambda x: x[0])

    # Train = all patches except val_patch/test_patch + first (1-val_frac) of val_patch
    train, val, test = [], [], []
    for p, rows in by_patch.items():
        mids = [mid for _, mid in rows]
        if p == args.test_patch:
            test = mids
        elif p == args.val_patch:
            k = int((1.0 - args.val_frac) * len(mids))
            train += mids[:k]
            val += mids[k:]
        else:
            train += mids

    def write(name, mids):
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8") as w:
            for mid in mids:
                w.write(str(mid) + "\n")
        print("Wrote", name, "=", len(mids))

    write("train.txt", train)
    write("val.txt", val)
    write("test.txt", test)

    # few-shot adaptation splits on test patch (time-ordered)
    for n in FEW_SHOT_NS:
        n_eff = min(n, len(test))
        write(f"test_calib_{n}.txt", test[:n_eff])
        write(f"test_eval_{n}.txt", test[n_eff:])

if __name__ == "__main__":
    main()
