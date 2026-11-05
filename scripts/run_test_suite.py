"""Execute Stage 10 Final Benchmark on Test-400 using frozen configuration (configs/final.yaml).

Systems evaluated:
1. closed_book_test: Closed-book parametric memory baseline (no retrieval)
2. standard_test: System A standard single-hop RAG baseline
3. ircot_test: System B IRCoT iterative retrieval comparator
4. erag_observe_test: System C Observational E-RAG (Gate G1 test verification)
5. erag_active_test: System C Proposed Full Active E-RAG
"""

import os
import subprocess
import sys
import time

PYTHON = sys.executable

TEST_EXPERIMENTS = [
    {
        "name": "standard_test",
        "cmd": [
            PYTHON, "scripts/run_experiment.py",
            "--config", "configs/base.yaml", "configs/final.yaml",
            "--set",
            "run.system=standard",
            "data.split=test",
            "run.name=standard_test",
        ],
    },
    {
        "name": "closed_book_test",
        "cmd": [
            PYTHON, "scripts/run_experiment.py",
            "--config", "configs/base.yaml", "configs/final.yaml",
            "--set",
            "run.system=closed_book",
            "data.split=test",
            "run.name=closed_book_test",
        ],
    },
    {
        "name": "ircot_test",
        "cmd": [
            PYTHON, "scripts/run_experiment.py",
            "--config", "configs/base.yaml", "configs/final.yaml",
            "--set",
            "run.system=ircot",
            "data.split=test",
            "run.name=ircot_test",
        ],
    },
    {
        "name": "erag_observe_test",
        "cmd": [
            PYTHON, "scripts/run_experiment.py",
            "--config", "configs/base.yaml", "configs/final.yaml",
            "--set",
            "run.system=erag",
            "erag.mode=observe",
            "data.split=test",
            "run.name=erag_observe_test",
        ],
    },
    {
        "name": "erag_active_test",
        "cmd": [
            PYTHON, "scripts/run_experiment.py",
            "--config", "configs/base.yaml", "configs/final.yaml",
            "--set",
            "run.system=erag",
            "erag.mode=active",
            "data.split=test",
            "run.name=erag_active_test",
        ],
    },
]


def is_complete(name: str, expected: int = 400) -> bool:
    path = os.path.join("results", f"{name}.jsonl")
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip()) >= expected


def main():
    print("=" * 70)
    print("STAGE 10: TEST-400 BENCHMARK SUITE (FROZEN CONFIG)")
    print("=" * 70)

    for exp in TEST_EXPERIMENTS:
        name = exp["name"]
        if is_complete(name):
            print(f"[*] {name} already complete (400 items). Skipping.")
            continue

        print(f"\n[>] Executing {name}...")
        t0 = time.time()
        res = subprocess.run(exp["cmd"], check=False)
        dur = time.time() - t0
        if res.returncode != 0:
            print(f"[!] {name} failed with exit code {res.returncode}")
            sys.exit(res.returncode)
        print(f"[+] {name} completed in {dur:.1f}s.")

    print("\n[>] Generating Test publication tables...")
    subprocess.run([PYTHON, "scripts/make_tables.py", "--split", "test"], check=False)

    print("\n[>] Generating Test publication figures...")
    subprocess.run([PYTHON, "scripts/make_figures.py", "--split", "test"], check=False)

    print("\n" + "=" * 70)
    print("Stage 10 Test-400 Benchmark Suite Completed Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

