"""Execute remaining dev-100 ablations and pooled retrieval setting sequentially.

Ablations:
- no_adaptive: Adaptive retrieval disabled (hop limit 0)
- no_fast: No-gap fast exit path disabled
- no_eig: Expected Information Gain query generator disabled (uses raw claim text)
- pooled: Full E-RAG under pooled retrieval setting (corpus shared across dev+test)
"""

import os
import subprocess
import sys
import time

PYTHON = sys.executable

ABLATIONS = [
    {
        "name": "erag_ablate_no_adaptive_dev",
        "cmd": [
            PYTHON, "scripts/run_experiment.py",
            "--config", "configs/base.yaml",
            "--set", "run.system=erag", "erag.mode=active",
            "run.name=erag_ablate_no_adaptive_dev",
            "data.split=dev",
            "erag.ablate=[no_adaptive]",
        ],
    },
    {
        "name": "erag_ablate_no_fast_dev",
        "cmd": [
            PYTHON, "scripts/run_experiment.py",
            "--config", "configs/base.yaml",
            "--set", "run.system=erag", "erag.mode=active",
            "run.name=erag_ablate_no_fast_dev",
            "data.split=dev",
            "erag.ablate=[no_fast]",
        ],
    },
    {
        "name": "erag_ablate_no_eig_dev",
        "cmd": [
            PYTHON, "scripts/run_experiment.py",
            "--config", "configs/base.yaml",
            "--set", "run.system=erag", "erag.mode=active",
            "run.name=erag_ablate_no_eig_dev",
            "data.split=dev",
            "erag.ablate=[no_eig]",
        ],
    },
    {
        "name": "erag_active_dev_pooled",
        "cmd": [
            PYTHON, "scripts/run_experiment.py",
            "--config", "configs/base.yaml",
            "--set", "run.system=erag", "erag.mode=active",
            "retrieval.setting=pooled",
            "run.name=erag_active_dev_pooled",
            "data.split=dev",
        ],
    },
]


def is_complete(name: str, expected_count: int = 100) -> bool:
    path = os.path.join("results", f"{name}.jsonl")
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        count = sum(1 for line in f if line.strip())
    return count >= expected_count


def main():
    print("=" * 70)
    print("DEV ABLATION & POOLED SUITE RUNNER")
    print("=" * 70)

    for exp in ABLATIONS:
        name = exp["name"]
        if is_complete(name):
            print(f"[*] {name} already complete (100 items). Skipping.")
            continue

        print(f"\n[>] Starting {name}...")
        t0 = time.time()
        res = subprocess.run(exp["cmd"], check=False)
        dur = time.time() - t0
        if res.returncode != 0:
            print(f"[!] {name} failed with exit code {res.returncode}")
            sys.exit(res.returncode)
        print(f"[+] {name} completed in {dur:.1f}s.")

    print("\n" + "=" * 70)
    print("All dev ablations and pooled runs finished successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

