import argparse, orjson
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    c = Counter()
    min_t, max_t = None, None
    n = 0
    with open(args.input, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            obj = orjson.loads(line)
            n += 1
            c[obj.get("patch")] += 1
            st = obj.get("start_time")
            if st is not None:
                min_t = st if min_t is None else min(min_t, st)
                max_t = st if max_t is None else max(max_t, st)

    print("Total matches:", n)
    print("Patch counts:", dict(c))
    print("Start_time range:", min_t, "->", max_t)

if __name__ == "__main__":
    main()
