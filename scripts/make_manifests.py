"""Build question manifests (debug, dev, test) from raw HotpotQA data with seeded stratification."""

import argparse
import collections
import hashlib
import json
import os
import random


def main():
    ap = argparse.ArgumentParser(description="Create frozen question manifests for E-RAG evaluation.")
    ap.add_argument("--raw", required=True, help="Path to raw hotpot_dev_distractor_v1.json")
    ap.add_argument("--debug-ids", required=True, help="Path to debug50_ids.txt")
    ap.add_argument("--n-dev", type=int, default=100, help="Number of questions in dev split")
    ap.add_argument("--n-test", type=int, default=400, help="Number of questions in test split")
    ap.add_argument("--seed", type=int, default=13, help="Random seed for deterministic split")
    ap.add_argument("--out", default="data/manifests", help="Output directory for manifests")
    a = ap.parse_args()

    with open(a.raw, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(a.debug_ids, "r", encoding="utf-8") as f:
        debug_ids = set(f.read().split())

    debug = [d for d in data if d["_id"] in debug_ids]
    pool = [d for d in data if d["_id"] not in debug_ids]
    assert len(debug) == len(debug_ids), "Some debug ids are not in the raw file"

    rng = random.Random(a.seed)
    strata = collections.defaultdict(list)
    for d in pool:
        strata[(d["type"], d["level"])].append(d)

    for v in strata.values():
        rng.shuffle(v)

    dev, test = [], []
    for key in sorted(strata):
        v = strata[key]
        nd = round(a.n_dev * len(v) / len(pool))
        nt = round(a.n_test * len(v) / len(pool))
        dev += v[:nd]
        test += v[nd : nd + nt]  # disjoint from dev by construction

    rng.shuffle(dev)
    rng.shuffle(test)
    dev, test = dev[: a.n_dev], test[: a.n_test]

    # Verify complete disjointness across all three splits
    debug_id_set = {d["_id"] for d in debug}
    dev_id_set = {d["_id"] for d in dev}
    test_id_set = {d["_id"] for d in test}

    assert len(debug_id_set & dev_id_set) == 0, "Overlap detected between debug and dev splits"
    assert len(debug_id_set & test_id_set) == 0, "Overlap detected between debug and test splits"
    assert len(dev_id_set & test_id_set) == 0, "Overlap detected between dev and test splits"

    os.makedirs(a.out, exist_ok=True)
    manifest = {
        "seed": a.seed,
        "raw_file": os.path.basename(a.raw),
        "raw_sha256": hashlib.sha256(open(a.raw, "rb").read()).hexdigest(),
    }

    for name, rows in (("debug", debug), ("dev", dev), ("test", test)):
        path = os.path.join(a.out, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        manifest[name] = {
            "n": len(rows),
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "by_type": dict(collections.Counter(r["type"] for r in rows)),
            "by_level": dict(collections.Counter(r.get("level", "unknown") for r in rows)),
        }

    manifest_path = os.path.join(a.out, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully generated manifests in {a.out}:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
