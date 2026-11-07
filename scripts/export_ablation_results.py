"""Export authoritative Dev-100 ablation results to CSV and Markdown."""

import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.metrics import brier, exact_match, f1_score, is_unknown


def load_run(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    results_dir = "results"
    out_dir = "research/methodology_audit"
    os.makedirs(out_dir, exist_ok=True)

    ablations = [
        ("Full E-RAG Active", "erag_active_dev.jsonl", "Baseline frozen system with all epistemic modules active"),
        ("w/o NLI Verification (no_nli)", "erag_ablate_no_nli_dev.jsonl", "Replaces DeBERTa NLI with neutral unresolved states"),
        ("w/o Epistemic Confidence (no_conf)", "erag_ablate_no_conf_dev.jsonl", "Replaces consensus confidence c0 with 1.0 (disables abstention)"),
        ("w/o Belief Propagation (no_prop)", "erag_ablate_no_prop_dev.jsonl", "Disables weakest-link DAG propagation (independent claim updating)"),
        ("w/o Adaptive Retrieval (no_adaptive)", "erag_ablate_no_adaptive_dev.jsonl", "Stops retrieval after initial hop (single-hop epistemic)"),
        ("w/o Fast Path (no_fast)", "erag_ablate_no_fast_dev.jsonl", "Forces retrieval to always execute maximum hops (no early exit)"),
        ("w/o EIG Query Selection (no_eig)", "erag_ablate_no_eig_dev.jsonl", "Uses raw claim string as search query instead of EIG proposal"),
        ("Pooled Setting (1,000 paras)", "erag_active_dev_pooled.jsonl", "Retrieval over full 1,000-paragraph corpus across all 100 dev questions"),
    ]

    rows = []
    for label, fname, desc in ablations:
        fpath = os.path.join(results_dir, fname)
        recs = load_run(fpath)
        if not recs:
            print(f"Warning: {fpath} not found")
            continue

        n = len(recs)
        ems = [r.get("em", 0.0) for r in recs]
        f1s = [r.get("f1", 0.0) for r in recs]
        recalls = [r.get("recall", 0.0) for r in recs if r.get("recall") is not None]
        all_golds = [1.0 if r.get("all_gold") else 0.0 for r in recs if r.get("all_gold") is not None]
        unknowns = [1.0 if r.get("unknown") else 0.0 for r in recs]
        
        answered = [r for r in recs if not r.get("unknown")]
        coverage = len(answered) / n if n > 0 else 0.0
        sel_em = np.mean([r.get("em", 0.0) for r in answered]) if answered else 0.0

        confs = [r.get("answer_conf", r.get("c0", 1.0)) for r in recs]
        brier_val = brier(confs, [int(e == 1.0) for e in ems])

        llm_calls = np.mean([r.get("llm_calls", 1.0) for r in recs])
        rets = np.mean([r.get("retrievals", 1.0) for r in recs])
        lats = [r.get("wall_latency", 0.0) for r in recs]
        tokens = [r.get("in_tokens", 0) + r.get("out_tokens", 0) for r in recs]

        rows.append({
            "Ablation Setting": label,
            "N": n,
            "EM": float(np.mean(ems)),
            "Token F1": float(np.mean(f1s)),
            "Gold Paragraph Recall": float(np.mean(recalls)) if recalls else np.nan,
            "All-Gold Rate": float(np.mean(all_golds)) if all_golds else np.nan,
            "Coverage": float(coverage),
            "Selective EM": float(sel_em),
            "Brier Score": float(brier_val),
            "Mean Retrievals/Q": float(rets),
            "Mean LLM Calls/Q": float(llm_calls),
            "Median Latency (s)": float(np.median(lats)),
            "Mean Total Tokens": float(np.mean(tokens)),
            "Description": desc,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print(df[["Ablation Setting", "EM", "Token F1", "Gold Paragraph Recall", "Coverage", "Selective EM", "Mean Retrievals/Q", "Median Latency (s)"]].to_string(index=False))


if __name__ == "__main__":
    main()

