"""Grid search over belief graph weights (w_s, w_c, propagate) for Gate G3 parameter freeze.

Specification: E-RAG Implementation Guide, Page 41-42 (Experiment E2).
Evaluates AUROC, ECE, and Brier score over all grid combinations with zero API/model calls.
"""

import argparse
import itertools
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.epistemic.belief_graph import BeliefGraph
from src.eval.metrics import auroc, brier, ece_by_level, exact_match


def run_grid(jsonl_path: str, use_draft: bool = True) -> List[Dict[str, Any]]:
    """Compute calibration metrics across hyperparameter grid."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    valid_rows = [r for r in rows if not r.get("failed") and r.get("trace")]
    print(f"Loaded {len(valid_rows)} valid evaluation rows from {jsonl_path}.")

    results = []
    # Grid specification: w_s in [0.5, 1, 2], w_c in [1, 2, 4], propagate in [True, False]
    grid = list(itertools.product([0.5, 1.0, 2.0], [1.0, 2.0, 4.0], [True, False]))

    for w_s, w_c, prop in grid:
        conf, ok = [], []
        for r in valid_rows:
            # Use the final trace step's belief graph
            final_trace = r["trace"][-1]
            graph_data = final_trace.get("graph", [])
            if not graph_data:
                continue

            g = BeliefGraph(w_s=w_s, w_c=w_c, propagate=prop)
            for c in graph_data:
                g.add(
                    text=c["text"],
                    answer_bearing=c["answer_bearing"],
                    depends_on=c["depends_on"],
                    c0=c["c0"],
                    ev={
                        "status": c["status"],
                        "conflict": c["conflict"],
                        "sup": c["sup"],
                        "con": c["con"],
                    },
                )
            g.update()
            ans_claim = g.answer_claim()
            conf.append(ans_claim.eff)

            if use_draft and "draft" in r:
                label = exact_match(r["draft"], r["gold"])
            else:
                label = r["em"]
            ok.append(label)

        auc = auroc(conf, ok)
        ece_val, _ = ece_by_level(conf, ok)
        br = brier(conf, ok)
        results.append({
            "auroc": auc,
            "ece": ece_val,
            "brier": br,
            "w_s": w_s,
            "w_c": w_c,
            "propagate": prop,
        })

    # Sort descending by AUROC, then ascending by ECE
    results.sort(key=lambda x: (x["auroc"], -x["ece"]), reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate belief graph weight grid.")
    parser.add_argument("path", help="Path to dev results JSONL (e.g. results/erag_observe_dev.jsonl or results/erag_active_dev.jsonl)")
    parser.add_argument("--use-answer", action="store_true", help="Evaluate against final answer EM instead of draft EM")
    args = parser.parse_args()

    results = run_grid(args.path, use_draft=not args.use_answer)

    print("\n" + "=" * 70)
    print("WEIGHT GRID SEARCH RESULTS (Sorted by AUROC descending)")
    print("=" * 70)
    print(f"{'Rank':<5} | {'AUROC':<7} | {'ECE':<7} | {'Brier':<7} | {'w_s':<5} | {'w_c':<5} | {'propagate':<10}")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        print(f"{i:<5} | {r['auroc']:<7.4f} | {r['ece']:<7.4f} | {r['brier']:<7.4f} | {r['w_s']:<5.1f} | {r['w_c']:<5.1f} | {str(r['propagate']):<10}")
    print("=" * 70)

    best = results[0]
    print(f"\nOptimal configuration for Gate G3: w_s={best['w_s']}, w_c={best['w_c']}, propagate={best['propagate']} (AUROC={best['auroc']:.4f}, ECE={best['ece']:.4f})")


if __name__ == "__main__":
    main()
