#!/usr/bin/env python3
"""Extract the soft-labelled positions into a mini shard + remapped jsonl.

The training loop indexes records 0..N-1 of its shard, so the soft-label
jsonl's shard_idx (positions in the SOURCE shard) must be remapped to the
mini shard's dense indices. Reads one or more make_soft_policy.py jsonl
parts (parallel workers), writes:
  DST            mini shard (records in sorted shard_idx order)
  DST.jsonl      merged jsonl with shard_idx remapped to mini indices
"""
import argparse, json, struct, sys

sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from format import MAGIC, VERSION, HEADER_SIZE, RECORD_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="source shard")
    ap.add_argument("--jsonl", required=True, help="comma-separated jsonl parts")
    ap.add_argument("--out", required=True, help="mini shard path")
    args = ap.parse_args()

    rows = {}
    for part in args.jsonl.split(","):
        with open(part) as f:
            for line in f:
                r = json.loads(line)
                rows[int(r["shard_idx"])] = r
    idxs = sorted(rows)
    print(f"{len(idxs)} labelled positions from {len(args.jsonl.split(','))} part(s)")

    remap = {old: new for new, old in enumerate(idxs)}
    with open(args.shard, "rb") as src, open(args.out, "wb") as dst:
        dst.write(struct.pack("<IHHQ", MAGIC, VERSION, HEADER_SIZE, len(idxs)))
        for old in idxs:
            src.seek(HEADER_SIZE + old * RECORD_SIZE)
            dst.write(src.read(RECORD_SIZE))
    with open(args.out + ".jsonl", "w") as out:
        for old in idxs:
            r = rows[old]
            r["shard_idx"] = remap[old]
            out.write(json.dumps(r) + "\n")
    print(f"wrote {args.out} ({len(idxs)} records) + remapped {args.out}.jsonl")


if __name__ == "__main__":
    main()
