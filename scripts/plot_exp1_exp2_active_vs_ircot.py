"""Generate dedicated Experiment 1 and Experiment 2 publication graphs strictly comparing E-RAG Active vs. IRCoT.

No extra variants: strictly E-RAG Active vs. IRCoT.

Outputs:
1. exp1_head_to_head_active_vs_ircot.png (Bar comparison + Question-level F1 scatter plot)
2. exp2_risk_coverage_and_calibration.png (Risk-Coverage Frontier + Confidence Calibration curves)
"""

import json
import os
import sys

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.eval.metrics import exact_match, f1_score


def main():
    res_dir = "research/erag_vs_ircot_study"
    artifact_dir = "C:/Users/hp/.gemini/antigravity/brain/f1e5200d-a581-4031-af73-9be369bc1639"
    os.makedirs(res_dir, exist_ok=True)

    ircot = [json.loads(l) for l in open("results/ircot_test.jsonl", encoding="utf-8") if l.strip()]
    act = [json.loads(l) for l in open("results/erag_active_test.jsonl", encoding="utf-8") if l.strip()]

    golds = [r["gold"] for r in act]

    # Metrics computation
    ircot_em = np.mean([r["em"] for r in ircot])
    ircot_f1 = np.mean([r["f1"] for r in ircot])
    ircot_recall = np.mean([r.get("recall", 0.0) for r in ircot])
    ircot_all_gold = np.mean([1.0 if r.get("all_gold") else 0.0 for r in ircot])
    ircot_cov = np.mean([r["answer"] != "unknown" for r in ircot])
    ircot_sel_em = np.mean([r["em"] for r in ircot if r["answer"] != "unknown"])

    act_em = np.mean([r["em"] for r in act])
    act_f1 = np.mean([r["f1"] for r in act])
    act_recall = np.mean([r.get("recall", 0.0) for r in act])
    act_all_gold = np.mean([1.0 if r.get("all_gold") else 0.0 for r in act])
    act_cov = np.mean([r["answer"] != "unknown" for r in act])
    act_sel_em = np.mean([r["em"] for r in act if r["answer"] != "unknown"])

    # -------------------------------------------------------------------------
    # EXPERIMENT 1 GRAPH: Head-to-Head Performance (Bars + Scatter)
    # -------------------------------------------------------------------------
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Panel 1: Bar Comparison Across Core Metrics
    metric_names = ["Exact Match", "Token F1", "Gold Recall", "All-Gold Rate", "Selective EM"]
    ircot_metrics = [ircot_em * 100, ircot_f1 * 100, ircot_recall * 100, ircot_all_gold * 100, ircot_sel_em * 100]
    act_metrics = [act_em * 100, act_f1 * 100, act_recall * 100, act_all_gold * 100, act_sel_em * 100]

    x = np.arange(len(metric_names))
    width = 0.35

    rects1 = ax1.bar(x - width/2, ircot_metrics, width, label="IRCoT Baseline", color="#ff7f0e", edgecolor="black", linewidth=0.8)
    rects2 = ax1.bar(x + width/2, act_metrics, width, label="E-RAG Active", color="#1f77b4", edgecolor="black", linewidth=0.8)

    ax1.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Core Metric Comparison (Test-400)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax1.legend(loc="upper left", fontsize=10)

    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")

    # Panel 2: Question-Level F1 Scatter Plot
    ircot_f1s = [r["f1"] for r in ircot]
    act_f1s = [r["f1"] for r in act]

    ties = sum(1 for i, a in zip(ircot_f1s, act_f1s) if abs(i - a) <= 1e-5)
    act_wins = sum(1 for i, a in zip(ircot_f1s, act_f1s) if a > i + 1e-5)
    ircot_wins = sum(1 for i, a in zip(ircot_f1s, act_f1s) if i > a + 1e-5)

    ax2.scatter(
        [i for i, a in zip(ircot_f1s, act_f1s) if abs(i - a) <= 1e-5],
        [a for i, a in zip(ircot_f1s, act_f1s) if abs(i - a) <= 1e-5],
        color="gray", alpha=0.4, s=35, label=f"Ties (n = {ties})"
    )
    ax2.scatter(
        [i for i, a in zip(ircot_f1s, act_f1s) if a > i + 1e-5],
        [a for i, a in zip(ircot_f1s, act_f1s) if a > i + 1e-5],
        color="#1f77b4", alpha=0.8, s=45, label=f"E-RAG Active Wins (n = {act_wins})"
    )
    ax2.scatter(
        [i for i, a in zip(ircot_f1s, act_f1s) if i > a + 1e-5],
        [a for i, a in zip(ircot_f1s, act_f1s) if i > a + 1e-5],
        color="#ff7f0e", alpha=0.8, s=45, label=f"IRCoT Wins (n = {ircot_wins})"
    )

    ax2.plot([-0.05, 1.05], [-0.05, 1.05], "k--", label="Parity (y = x)")
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("IRCoT Token F1 Score", fontsize=11, fontweight="bold")
    ax2.set_ylabel("E-RAG Active Token F1 Score", fontsize=11, fontweight="bold")
    ax2.set_title("B. Question-Level F1 Scatter (N = 400)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    p1 = os.path.join(res_dir, "exp1_head_to_head_active_vs_ircot.png")
    plt.savefig(p1)
    plt.savefig(os.path.join(artifact_dir, "exp1_head_to_head_active_vs_ircot.png"))
    plt.close()

    # -------------------------------------------------------------------------
    # EXPERIMENT 2 GRAPH: Selective Risk-Coverage & Calibration
    # -------------------------------------------------------------------------
    fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Panel 1: Selective Accuracy vs. Coverage Frontier
    # Sweep theta from 0.0 to 0.8
    ths = np.linspace(0.0, 0.75, 30)
    cov_curve, sel_em_curve = [], []

    for t in ths:
        sim_preds = ["unknown" if float(r.get("answer_conf", 0.0)) < t else r["answer"] for r in act]
        cov = np.mean([p != "unknown" for p in sim_preds])
        ems = [exact_match(p, g) for p, g in zip(sim_preds, golds)]
        sel_em = np.mean([e for p, e in zip(sim_preds, ems) if p != "unknown"]) if cov > 0 else 0.0
        cov_curve.append(cov)
        sel_em_curve.append(sel_em)

    bx1.plot([c * 100 for c in cov_curve], [s * 100 for s in sel_em_curve], marker="o", color="#1f77b4", linewidth=2.5, markersize=5, label="E-RAG Active Epistemic Frontier")
    
    # E-RAG Active frozen operating point (theta_abs = 0.40)
    bx1.scatter([act_cov * 100], [act_sel_em * 100], color="#1f77b4", s=200, edgecolors="black", linewidths=1.5, zorder=6, label=f"E-RAG Active Operating Point (Cov={act_cov*100:.1f}%, Sel-EM={act_sel_em*100:.1f}%)")

    # IRCoT static point
    bx1.scatter([ircot_cov * 100], [ircot_sel_em * 100], color="#ff7f0e", marker="s", s=200, edgecolors="black", linewidths=1.5, zorder=6, label=f"IRCoT Comparator (Cov={ircot_cov*100:.1f}%, Sel-EM={ircot_sel_em*100:.1f}%)")
    bx1.axhline(ircot_sel_em * 100, color="#ff7f0e", linestyle="--", linewidth=1.8, alpha=0.8)

    bx1.set_xlabel("Coverage (% of Questions Answered)", fontsize=11, fontweight="bold")
    bx1.set_ylabel("Selective Exact Match (%)", fontsize=11, fontweight="bold")
    bx1.set_title("A. Selective Risk-Coverage Frontier", fontsize=12, fontweight="bold")
    bx1.grid(True, linestyle="--", alpha=0.5)
    bx1.set_ylim(45, 70)
    bx1.legend(loc="lower left", fontsize=9.5)

    # Panel 2: Confidence vs. Accuracy Calibration Comparison
    # Discrete confidence levels: E-RAG continuous conf binned into 5 bins vs IRCoT
    bins = [0.0, 0.4, 0.6, 0.8, 1.01]
    bin_labels = ["0.0-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    
    act_confs = np.array([float(r.get("answer_conf", 0.0)) for r in act])
    act_ems = np.array([r["em"] for r in act])

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(len(bins) - 1):
        mask = (act_confs >= bins[i]) & (act_confs < bins[i+1])
        if np.sum(mask) > 0:
            bin_accs.append(np.mean(act_ems[mask]) * 100)
            bin_confs.append(np.mean(act_confs[mask]) * 100)
            bin_counts.append(np.sum(mask))
        else:
            bin_accs.append(0.0)
            bin_confs.append((bins[i] + bins[i+1]) / 2 * 100)
            bin_counts.append(0)

    xb = np.arange(len(bin_labels))
    bx2.bar(xb - 0.15, bin_confs, 0.3, label="Predicted Confidence (%)", color="#aec7e8", edgecolor="black")
    bx2.bar(xb + 0.15, bin_accs, 0.3, label="Actual Accuracy (EM %)", color="#1f77b4", edgecolor="black")

    bx2.set_xticks(xb)
    bx2.set_xticklabels(bin_labels, fontsize=10, fontweight="bold")
    bx2.set_xlabel("Epistemic Confidence Bin", fontsize=11, fontweight="bold")
    bx2.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    bx2.set_title("B. E-RAG Active Confidence Calibration", fontsize=12, fontweight="bold")
    bx2.set_ylim(0, 105)
    bx2.grid(True, linestyle=":", alpha=0.5, axis="y")
    bx2.legend(loc="upper left", fontsize=9.5)

    for i, count in enumerate(bin_counts):
        bx2.annotate(f"n={count}", xy=(xb[i], 5), ha="center", fontsize=8.5, color="white", fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="#333333", ec="none"))

    plt.tight_layout()
    p2 = os.path.join(res_dir, "exp2_risk_coverage_and_calibration.png")
    plt.savefig(p2)
    plt.savefig(os.path.join(artifact_dir, "exp2_risk_coverage_and_calibration.png"))
    plt.close()

    print("[+] Saved exp1_head_to_head_active_vs_ircot.png and exp2_risk_coverage_and_calibration.png")


if __name__ == "__main__":
    main()

