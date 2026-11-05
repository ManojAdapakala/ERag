"""Generate publication-ready LaTeX, Markdown, and CSV tables from experimental results."""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.metrics import brier, ece_by_level
from src.eval.stats import holm, mcnemar_exact, paired_bootstrap


def load_run(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load JSONL run records."""
    if not os.path.exists(jsonl_path):
        return []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_system_row(name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate row metrics for a single system run."""
    if not records:
        return {"System": name, "Status": "MISSING"}
    
    n = len(records)
    ems = [r.get("em", 0.0) for r in records]
    f1s = [r.get("f1", 0.0) for r in records]
    recalls = [r.get("recall", 0.0) for r in records if r.get("recall") is not None and not np.isnan(r.get("recall"))]
    all_golds = [1.0 if r.get("all_gold") else 0.0 for r in records if r.get("all_gold") is not None]
    unknowns = [1.0 if r.get("unknown") else 0.0 for r in records]
    
    answered = [r for r in records if not r.get("unknown")]
    selective_em = np.mean([r.get("em", 0.0) for r in answered]) if answered else 0.0
    coverage = len(answered) / n if n > 0 else 0.0
    
    confs = [r.get("answer_conf", r.get("c0", 1.0)) for r in records]
    ece, _ = ece_by_level(confs, [int(em == 1.0) for em in ems])
    brier_val = brier(confs, [int(em == 1.0) for em in ems])
    
    llm_calls = np.mean([r.get("llm_calls", 1.0) for r in records])
    retrievals = np.mean([r.get("retrievals", 1.0) for r in records])
    
    return {
        "System": name,
        "N": n,
        "EM": float(np.mean(ems)),
        "F1": float(np.mean(f1s)),
        "Recall": float(np.mean(recalls)) if recalls else 0.0,
        "All-Gold": float(np.mean(all_golds)) if all_golds else 0.0,
        "Unknown %": float(np.mean(unknowns)) * 100.0,
        "Coverage": coverage,
        "Selective EM": selective_em,
        "ECE": ece,
        "Brier": brier_val,
        "LLM Calls/Q": llm_calls,
        "Retrievals/Q": retrievals,
    }


def make_main_table(results_dir: str = "results", split: str = "debug") -> pd.DataFrame:
    """Generate Table 1: Main benchmark performance comparison across systems."""
    systems = [
        ("Closed-Book", f"closed_book_{split}.jsonl"),
        ("System A (Standard RAG)", f"standard_{split}.jsonl"),
        ("System B (IRCoT)", f"ircot_{split}.jsonl"),
        ("System C (E-RAG Observe)", f"erag_observe_{split}.jsonl"),
        ("System C (E-RAG Active)", f"erag_active_{split}.jsonl"),
    ]
    rows = []
    for label, fname in systems:
        path = os.path.join(results_dir, fname)
        recs = load_run(path)
        rows.append(compute_system_row(label, recs))
    
    return pd.DataFrame(rows)


def make_ablation_table(results_dir: str = "results", split: str = "dev") -> pd.DataFrame:
    """Generate Table 2: Ablation study showing contribution of each epistemic component."""
    variants = [
        ("Full E-RAG", f"erag_active_{split}.jsonl"),
        ("w/o NLI Verification (no_nli)", f"erag_ablate_no_nli_{split}.jsonl"),
        ("w/o Epistemic Confidence (no_conf)", f"erag_ablate_no_conf_{split}.jsonl"),
        ("w/o Belief Propagation (no_prop)", f"erag_ablate_no_prop_{split}.jsonl"),
        ("w/o Adaptive Retrieval (no_adaptive)", f"erag_ablate_no_adaptive_{split}.jsonl"),
        ("w/o Fast Path (no_fast)", f"erag_ablate_no_fast_{split}.jsonl"),
        ("w/o EIG Query Selection (no_eig)", f"erag_ablate_no_eig_{split}.jsonl"),
    ]
    rows = []
    for label, fname in variants:
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            recs = load_run(path)
            rows.append(compute_system_row(label, recs))
    return pd.DataFrame(rows)


def make_significance_table(
    baseline_path: str,
    comparator_paths: Dict[str, str],
) -> pd.DataFrame:
    """Generate Table 3: Statistical significance test table with bootstrap CIs and McNemar exact tests."""
    base_recs = {r["id"]: r for r in load_run(baseline_path)}
    if not base_recs:
        return pd.DataFrame()
    
    rows = []
    p_values = []
    for name, path in comparator_paths.items():
        comp_recs = {r["id"]: r for r in load_run(path)}
        common_ids = [qid for qid in base_recs if qid in comp_recs]
        if not common_ids:
            continue
        
        base_em = [base_recs[qid].get("em", 0.0) for qid in common_ids]
        comp_em = [comp_recs[qid].get("em", 0.0) for qid in common_ids]
        base_f1 = [base_recs[qid].get("f1", 0.0) for qid in common_ids]
        comp_f1 = [comp_recs[qid].get("f1", 0.0) for qid in common_ids]
        
        d_em, em_lo, em_hi = paired_bootstrap(base_em, comp_em)
        d_f1, f1_lo, f1_hi = paired_bootstrap(base_f1, comp_f1)
        
        b_bin = [bool(x == 1.0) for x in base_em]
        c_bin = [bool(x == 1.0) for x in comp_em]
        mcn = mcnemar_exact(b_bin, c_bin)
        p_val = mcn["p"]
        b = mcn["a_only"]
        c = mcn["b_only"]
        p_values.append(p_val)
        
        rows.append({
            "Comparison vs Standard": name,
            "N": len(common_ids),
            "Delta EM": d_em,
            "EM 95% CI": f"[{em_lo:+.3f}, {em_hi:+.3f}]",
            "Delta F1": d_f1,
            "F1 95% CI": f"[{f1_lo:+.3f}, {f1_hi:+.3f}]",
            "Discordant (B/C)": f"{b}/{c}",
            "p-value (McNemar)": p_val,
        })
    
    if rows:
        adj_p = holm(p_values)
        for r, ap in zip(rows, adj_p):
            r["p-adj (Holm)"] = ap
            r["Significant (p<0.05)"] = "Yes" if ap < 0.05 else "No"
            
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as standard Markdown table without external tabulate dependency."""
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark and analysis tables.")
    parser.add_argument("--results-dir", default="results", help="Directory with experiment jsonl files")
    parser.add_argument("--split", default="debug", help="Split name to analyze (debug, dev, test)")
    parser.add_argument("--out-dir", default="research/tables", help="Output directory for generated tables")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"=== Generating Table 1: Main Performance ({args.split}) ===")
    df_main = make_main_table(args.results_dir, args.split)
    print(df_main.to_string(index=False))
    df_main.to_csv(os.path.join(args.out_dir, f"table1_main_{args.split}.csv"), index=False)
    with open(os.path.join(args.out_dir, f"table1_main_{args.split}.md"), "w", encoding="utf-8") as f:
        f.write(df_to_markdown(df_main))

    df_abl = make_ablation_table(args.results_dir, args.split)
    if not df_abl.empty and len(df_abl) > 1:
        print(f"\n=== Generating Table 2: Ablation Study ({args.split}) ===")
        print(df_abl.to_string(index=False))
        df_abl.to_csv(os.path.join(args.out_dir, f"table2_ablation_{args.split}.csv"), index=False)
        with open(os.path.join(args.out_dir, f"table2_ablation_{args.split}.md"), "w", encoding="utf-8") as f:
            f.write(df_to_markdown(df_abl))
    
    
    base_file = os.path.join(args.results_dir, f"standard_{args.split}.jsonl")
    if os.path.exists(base_file):
        print(f"\n=== Generating Table 3: Statistical Significance ({args.split}) ===")
        comparators = {
            "IRCoT": os.path.join(args.results_dir, f"ircot_{args.split}.jsonl"),
            "E-RAG Observe": os.path.join(args.results_dir, f"erag_observe_{args.split}.jsonl"),
            "E-RAG Active": os.path.join(args.results_dir, f"erag_active_{args.split}.jsonl"),
        }
        df_sig = make_significance_table(base_file, {k: v for k, v in comparators.items() if os.path.exists(v)})
        if not df_sig.empty:
            print(df_sig.to_string(index=False))
            df_sig.to_csv(os.path.join(args.out_dir, f"table3_significance_{args.split}.csv"), index=False)
            with open(os.path.join(args.out_dir, f"table3_significance_{args.split}.md"), "w", encoding="utf-8") as f:
                f.write(df_to_markdown(df_sig))
    
    print(f"\nTables successfully saved in {args.out_dir}/")


if __name__ == "__main__":
    main()

