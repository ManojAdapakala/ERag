"""Generate publication-ready figures for E-RAG research paper and reports."""

import argparse
import json
import os
import sys
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.metrics import ece_by_level


def set_paper_style():
    """Apply clean, academic styling for publication figures."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.autolayout": True,
    })


def load_run(jsonl_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(jsonl_path):
        return []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_reliability_diagram(records: List[Dict[str, Any]], out_path: str, title: str = "Reliability Diagram"):
    """Plot empirical accuracy vs predicted epistemic confidence with calibration gap."""
    if not records:
        return
    
    confs = [r.get("answer_conf", r.get("c0", 1.0)) for r in records]
    accs = [1 if r.get("em", 0.0) == 1.0 else 0 for r in records]
    ece, rows = ece_by_level(confs, accs)
    
    if not rows:
        return
    
    levels = [r["confidence"] for r in rows]
    empirical = [r["accuracy"] for r in rows]
    counts = [r["n"] for r in rows]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    
    # Upper panel: Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect Calibration")
    ax1.plot(levels, empirical, "o-", color="#1f77b4", linewidth=2, markersize=8, label=f"E-RAG (ECE = {ece:.3f})")
    ax1.fill_between(levels, levels, empirical, color="#1f77b4", alpha=0.15, label="Calibration Gap")
    
    ax1.set_ylabel("Empirical Accuracy (EM)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Lower panel: Confidence sample distribution histogram
    ax2.bar(levels, counts, width=0.1, color="#7f7f7f", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Predicted Epistemic Confidence")
    ax2.set_ylabel("Count")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Reliability Diagram to {out_path}")


def plot_risk_coverage_curve(runs: Dict[str, List[Dict[str, Any]]], out_path: str):
    """Plot Selective EM as a function of coverage (Risk-Coverage trade-off)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    colors = {"Standard RAG": "#d62728", "IRCoT": "#ff7f0e", "E-RAG Active": "#2ca02c", "E-RAG Observe": "#1f77b4"}
    
    for label, records in runs.items():
        if not records:
            continue
        confs = np.array([r.get("answer_conf", r.get("c0", 1.0)) for r in records])
        ems = np.array([r.get("em", 0.0) for r in records])
        
        thresholds = np.linspace(0.0, 1.0, 50)
        covs = []
        sel_accs = []
        
        for t in thresholds:
            mask = confs >= t
            cov = np.mean(mask)
            sel_acc = np.mean(ems[mask]) if np.any(mask) else 1.0
            covs.append(cov)
            sel_accs.append(sel_acc)
            
        ax.plot(covs, sel_accs, label=label, color=colors.get(label, None), linewidth=2)
    
    ax.set_xlabel("Coverage (Fraction of Questions Answered)")
    ax.set_ylabel("Selective Accuracy (EM on Answered)")
    ax.set_title("Selective Risk-Coverage Trade-off")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.2, 1.02)
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Risk-Coverage Curve to {out_path}")


def plot_pareto_frontier(summary_table_path: str, out_path: str):
    """Plot Efficiency (API calls) vs Quality (EM) Pareto frontier."""
    if not os.path.exists(summary_table_path):
        return
    import pandas as pd
    df = pd.read_csv(summary_table_path)
    if "EM" not in df.columns or "LLM Calls/Q" not in df.columns:
        return
        
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for _, row in df.iterrows():
        ax.scatter(row["LLM Calls/Q"], row["EM"], s=120, zorder=5)
        ax.annotate(
            row["System"],
            (row["LLM Calls/Q"], row["EM"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
        )
        
    ax.set_xlabel("Average LLM Calls per Question")
    ax.set_ylabel("Exact Match (EM)")
    ax.set_title("Quality vs. Computation Pareto Curve")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Pareto Frontier to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate figures from experiment outputs.")
    parser.add_argument("--results-dir", default="results", help="Directory with jsonl result files")
    parser.add_argument("--split", default="debug", help="Split name to plot (debug, dev, test)")
    parser.add_argument("--out-dir", default="research/figures", help="Output directory for generated plots")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    set_paper_style()
    
    # 1. Reliability Diagram for E-RAG
    erag_recs = load_run(os.path.join(args.results_dir, f"erag_observe_{args.split}.jsonl"))
    if not erag_recs:
        erag_recs = load_run(os.path.join(args.results_dir, f"erag_active_{args.split}.jsonl"))
    if erag_recs:
        plot_reliability_diagram(
            erag_recs,
            os.path.join(args.out_dir, f"fig1_calibration_{args.split}.png"),
            title=f"E-RAG Epistemic Calibration Curve ({args.split})",
        )
    
    # 2. Risk-Coverage Curves
    runs = {
        "Standard RAG": load_run(os.path.join(args.results_dir, f"standard_{args.split}.jsonl")),
        "IRCoT": load_run(os.path.join(args.results_dir, f"ircot_{args.split}.jsonl")),
        "E-RAG Active": load_run(os.path.join(args.results_dir, f"erag_active_{args.split}.jsonl")),
        "E-RAG Observe": load_run(os.path.join(args.results_dir, f"erag_observe_{args.split}.jsonl")),
    }
    plot_risk_coverage_curve(runs, os.path.join(args.out_dir, f"fig2_risk_coverage_{args.split}.png"))
    
    # 3. Pareto Frontier
    summary_csv = os.path.join("research/tables", f"table1_main_{args.split}.csv")
    if os.path.exists(summary_csv):
        plot_pareto_frontier(summary_csv, os.path.join(args.out_dir, f"fig3_pareto_{args.split}.png"))


if __name__ == "__main__":
    main()

