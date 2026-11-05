"""Execute Stage 9 Episodic Memory suite (Experiment E5).

Runs:
1. stream_nomem: E-RAG on the 30-question clustered stream with memory disabled.
2. stream_mem: E-RAG on the 30-question stream with episodic memory enabled.
3. stream_planted: E-RAG on the stream with 5 synthetic wrong beliefs planted to evaluate override dynamics.
"""

import os
import subprocess
import sys
import time

PYTHON = sys.executable

EXPERIMENTS = [
    {
        "name": "stream_nomem",
        "cmd": [
            PYTHON, "scripts/run_experiment.py",
            "--config", "configs/base.yaml", "configs/final.yaml",
            "--set",
            "run.system=erag",
            "erag.mode=active",
            "data.split=stream",
            "run.name=stream_nomem",
        ],
    },
    {
        "name": "stream_mem",
        "cmd": [
            PYTHON, "scripts/run_experiment.py",
            "--config", "configs/base.yaml", "configs/final.yaml",
            "--set",
            "run.system=erag",
            "erag.mode=active",
            "memory.enabled=true",
            "data.split=stream",
            "run.name=stream_mem",
        ],
    },
    {
        "name": "stream_planted",
        "cmd": [
            PYTHON, "scripts/run_experiment.py",
            "--config", "configs/base.yaml", "configs/final.yaml",
            "--set",
            "run.system=erag",
            "erag.mode=active",
            "memory.enabled=true",
            "memory.plant=data/manifests/planted.json",
            "data.split=stream",
            "run.name=stream_planted",
        ],
    },
]


def is_complete(name: str, expected: int = 30) -> bool:
    path = os.path.join("results", f"{name}.jsonl")
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip()) >= expected


def main():
    print("=" * 70)
    print("STAGE 9: MEMORY SUITE EXECUTION (EXPERIMENT E5)")
    print("=" * 70)

    for exp in EXPERIMENTS:
        name = exp["name"]
        if is_complete(name):
            print(f"[*] {name} already complete (30 items). Skipping.")
            continue

        print(f"\n[>] Executing {name}...")
        t0 = time.time()
        res = subprocess.run(exp["cmd"], check=False)
        dur = time.time() - t0
        if res.returncode != 0:
            print(f"[!] {name} failed with exit code {res.returncode}")
            sys.exit(res.returncode)
        print(f"[+] {name} completed in {dur:.1f}s.")

    print("\n[>] Running memory evaluation analysis...")
    subprocess.run([PYTHON, "scripts/eval_memory.py"], check=False)
    print("\n" + "=" * 70)
    print("Stage 9 Memory Suite Completed Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

