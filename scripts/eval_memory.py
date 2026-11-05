"""Analyze Verified-Belief Episodic Memory performance and override dynamics (Experiment E5).

Evaluates:
1. Memory Reuse Rate (% of questions utilizing verified prior beliefs)
2. External Retrievals Saved (retrievals_nomem - retrievals_mem)
3. Accuracy & F1 impact (EM, F1 comparison between stream_nomem and stream_mem)
4. Planted Wrong-Belief Overrides (% of synthetic erroneous beliefs detected & purged)
5. False Repetition Avoidance (preventing propagation of obsolete or erroneous beliefs)
"""

import json
import os
import sys
from typing import Any, Dict, List
import numpy as np


def load_run(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze_memory(
    nomem_path: str = "results/stream_nomem.jsonl",
    mem_path: str = "results/stream_mem.jsonl",
    planted_path: str = "results/stream_planted.jsonl",
    plant_def_path: str = "data/manifests/planted.json",
):
    print("=" * 70)
    print("EXPERIMENT E5: VERIFIED-BELIEF EPISODIC MEMORY BENCHMARK")
    print("=" * 70)

    nomem = load_run(nomem_path)
    mem = load_run(mem_path)
    planted = load_run(planted_path)

    if not nomem or not mem:
        print("[!] stream_nomem or stream_mem results not found yet.")
        return

    n = len(mem)
    nomem_em = np.mean([r.get("em", 0.0) for r in nomem])
    mem_em = np.mean([r.get("em", 0.0) for r in mem])
    nomem_f1 = np.mean([r.get("f1", 0.0) for r in nomem])
    mem_f1 = np.mean([r.get("f1", 0.0) for r in mem])

    nomem_ret = np.mean([r.get("retrievals", 1.0) for r in nomem])
    mem_ret = np.mean([r.get("retrievals", 1.0) for r in mem])

    reused = sum(1 for r in mem if r.get("mem_used", False))
    reuse_rate = reused / n if n > 0 else 0.0

    print(f"\n1. Memory Utilization & Efficiency:")
    print(f"   Questions in stream:       {n}")
    print(f"   Memory Reuse Rate:         {reuse_rate * 100:.1f}% ({reused}/{n} questions)")
    print(f"   External Retrievals/Q (No-Mem): {nomem_ret:.2f}")
    print(f"   External Retrievals/Q (Mem):    {mem_ret:.2f}")
    print(f"   Retrieval Reduction:            {((nomem_ret - mem_ret) / max(nomem_ret, 1e-6)) * 100:+.1f}%")

    print(f"\n2. Accuracy Impact:")
    print(f"   EM:  No-Mem = {nomem_em:.4f} | Mem = {mem_em:.4f} (Delta: {mem_em - nomem_em:+.4f})")
    print(f"   F1:  No-Mem = {nomem_f1:.4f} | Mem = {mem_f1:.4f} (Delta: {mem_f1 - nomem_f1:+.4f})")

    if planted and os.path.exists(plant_def_path):
        with open(plant_def_path, "r", encoding="utf-8") as f:
            plant_defs = json.load(f)
        n_planted = len(plant_defs)

        all_overrides = []
        for r in planted:
            for ev in r.get("mem_events", []):
                if ev.get("event") == "overridden":
                    all_overrides.append(ev)

        unique_overridden = set(ev.get("belief", "") for ev in all_overrides)

        print(f"\n3. Planted Wrong-Belief Override Dynamics:")
        print(f"   Synthetic Wrong Beliefs:   {n_planted}")
        print(f"   Total Override Events:     {len(all_overrides)}")
        print(f"   Unique Beliefs Purged:     {len(unique_overridden)} / {n_planted}")
        print(f"   Override Success Rate:     {(len(unique_overridden) / max(n_planted, 1)) * 100:.1f}%")
        print("\n   Override Log:")
        for ev in all_overrides:
            print(f"     [-] Purged: \"{ev.get('belief')}\"")
            print(f"         Contradicted by fresh passage: {ev.get('by')}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    analyze_memory()

