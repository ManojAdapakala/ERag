"""Generate dedicated 1-vs-1 comparison figures for IRCoT vs E-RAG Active (Part K3).

Outputs both high-resolution PNG (300 DPI) and vector PDF into:
- research/methodology_audit/ircot_vs_erag_active_plots/
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Academic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})


def save_fig(fig, out_dir: str, base_name: str):
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{base_name}.png")
    pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path} and {pdf_path}")


def plot_k3_1_and_2_f1_diff(df_paired: pd.DataFrame, out_dir: str):
    """K3.1 & K3.2: Histogram and ECDF of per-question F1 difference (Active - IRCoT)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    diffs = df_paired["delta_f1"].values
    mean_diff = np.mean(diffs)
    median_diff = np.median(diffs)

    # Panel 1: Histogram
    bins = np.linspace(-1.0, 1.0, 21)
    n_pos = np.sum(diffs > 0)
    n_neg = np.sum(diffs < 0)
    n_zero = np.sum(diffs == 0)

    ax1.hist(diffs, bins=bins, color="#2ca02c", edgecolor="black", alpha=0.75)
    ax1.axvline(0.0, color="black", linestyle="--", linewidth=1.5, label="Zero Difference (Tie)")
    ax1.axvline(mean_diff, color="#d62728", linestyle="-", linewidth=2.0, label=rf"Mean $\Delta = {mean_diff:+.3f}$")
    ax1.set_title("Distribution of Per-Question F1 Difference\n$(\\mathrm{F1}_{\\mathrm{Active}} - \\mathrm{F1}_{\\mathrm{IRCoT}})$", fontweight="bold")
    ax1.set_xlabel(r"F1 Difference ($\Delta$ F1)")
    ax1.set_ylabel("Question Count ($N=400$)")
    ax1.text(0.03, 0.95, f"Active Wins: {n_pos}\nTies: {n_zero}\nIRCoT Wins: {n_neg}", transform=ax1.transAxes, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax1.legend(loc="upper right")

    # Panel 2: ECDF
    sorted_diffs = np.sort(diffs)
    y_vals = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs)
    ax2.step(sorted_diffs, y_vals, where="post", color="#1f77b4", linewidth=2.2, label=rf"ECDF (Median={median_diff:+.3f})")
    ax2.axvline(0.0, color="black", linestyle="--", alpha=0.7)
    ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.6)
    ax2.set_title(r"Empirical Cumulative Distribution of $\Delta$ F1", fontweight="bold")
    ax2.set_xlabel(r"F1 Difference ($\Delta$ F1)")
    ax2.set_ylabel(r"Proportion of Questions ($\Delta \mathrm{F1} \leq x$)")
    ax2.set_xlim(-1.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(loc="lower right")

    fig.suptitle("HotpotQA Test-400: Paired F1 Improvement Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_k3_1_and_2_f1_difference_analysis")


def plot_k3_3_f1_scatter(df_paired: pd.DataFrame, out_dir: str):
    """K3.3: Scatter plot of IRCoT F1 vs E-RAG Active F1 with y=x reference line."""
    fig, ax = plt.subplots(figsize=(7.5, 7))

    x = df_paired["ircot_f1"].values
    y = df_paired["active_f1"].values

    # Categorize pairs
    active_win = y > x
    ircot_win = x > y
    ties = x == y

    # Add slight jitter for visualization of point density
    rng = np.random.RandomState(42)
    jitter_x = x + rng.uniform(-0.015, 0.015, size=len(x))
    jitter_y = y + rng.uniform(-0.015, 0.015, size=len(y))

    ax.scatter(jitter_x[ties], jitter_y[ties], color="#7f7f7f", alpha=0.45, s=28, label=f"Ties ($n={np.sum(ties)}$)")
    ax.scatter(jitter_x[active_win], jitter_y[active_win], color="#2ca02c", alpha=0.75, s=35, label=f"E-RAG Active Wins ($n={np.sum(active_win)}$)")
    ax.scatter(jitter_x[ircot_win], jitter_y[ircot_win], color="#d62728", alpha=0.75, s=35, label=f"IRCoT Wins ($n={np.sum(ircot_win)}$)")

    # Reference line y = x
    ax.plot([-0.05, 1.05], [-0.05, 1.05], "k--", linewidth=1.5, label="Parity ($y = x$)")

    ax.set_title("Question-Level F1: IRCoT vs E-RAG Active ($N=400$)", fontweight="bold")
    ax.set_xlabel("IRCoT Token F1 Score")
    ax.set_ylabel("E-RAG Active Token F1 Score")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    save_fig(fig, out_dir, "fig_k3_3_f1_scatter_plot")


def plot_k3_4_correctness_transition(df_paired: pd.DataFrame, out_dir: str):
    """K3.4: Correctness transition chart (paired 2x2 categories)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ircot_em = df_paired["ircot_em"].values
    active_em = df_paired["active_em"].values

    both_correct = np.sum((active_em == 1) & (ircot_em == 1))
    active_only = np.sum((active_em == 1) & (ircot_em == 0))
    ircot_only = np.sum((active_em == 0) & (ircot_em == 1))
    both_incorrect = np.sum((active_em == 0) & (ircot_em == 0))

    categories = [
        "Both Correct\n($n=144$)",
        "Active Only Correct\n($n=37$)",
        "IRCoT Only Correct\n($n=26$)",
        "Both Incorrect\n($n=193$)",
    ]
    counts = [both_correct, active_only, ircot_only, both_incorrect]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#7f7f7f"]

    bars = ax.bar(categories, counts, color=colors, edgecolor="black", width=0.55, alpha=0.9)
    ax.set_title("Paired Correctness Contingency Breakdown ($N=400$)", fontweight="bold")
    ax.set_ylabel("Question Count")
    ax.set_ylim(0, 230)

    for bar, count in zip(bars, counts):
        pct = count / 400.0 * 100.0
        ax.text(bar.get_x() + bar.get_width() / 2, count + 4, f"{count} ({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")

    ax.text(0.03, 0.92, f"McNemar Discordant Pairs: 37 vs 26\nNet Accuracy Gain: +11 (+2.75%)\nExact McNemar p-value: 0.2072", transform=ax.transAxes, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    plt.tight_layout()
    save_fig(fig, out_dir, "fig_k3_4_correctness_transition")


def plot_k3_5_recall_diff(df_paired: pd.DataFrame, out_dir: str):
    """K3.5: Gold paragraph recall difference distribution."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    rec_diff = df_paired["delta_recall"].values
    mean_rec_diff = np.mean(rec_diff)

    bins = np.linspace(-1.0, 1.0, 11)
    ax.hist(rec_diff, bins=bins, color="#1f77b4", edgecolor="black", alpha=0.8)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5, label="Parity (0.0)")
    ax.axvline(mean_rec_diff, color="#d62728", linestyle="-", linewidth=2.0, label=rf"Mean $\Delta = {mean_rec_diff:+.3f}$")

    n_active_more = np.sum(rec_diff > 0)
    n_equal = np.sum(rec_diff == 0)
    n_ircot_more = np.sum(rec_diff < 0)

    ax.set_title("Gold Paragraph Retrieval Recall Gain\n$(\\mathrm{Recall}_{\\mathrm{Active}} - \\mathrm{Recall}_{\\mathrm{IRCoT}})$", fontweight="bold")
    ax.set_xlabel(r"Recall Difference ($\Delta$ Recall)")
    ax.set_ylabel("Question Count ($N=400$)")
    ax.text(0.03, 0.95, f"Active Recalls More: {n_active_more}\nIdentical Recall: {n_equal}\nIRCoT Recalls More: {n_ircot_more}", transform=ax.transAxes, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax.legend(loc="upper right")

    plt.tight_layout()
    save_fig(fig, out_dir, "fig_k3_5_recall_diff_histogram")


def plot_k3_6_and_7_tradeoffs(df_paired: pd.DataFrame, out_dir: str):
    """K3.6 & K3.7: Trade-off comparisons (Latency vs F1, and Tokens vs Recall)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Wall Latency vs F1 Score
    ax1.scatter(df_paired["ircot_wall_latency"], df_paired["ircot_f1"], color="#ff7f0e", alpha=0.55, s=28, label=f"IRCoT (Med Lat={np.median(df_paired['ircot_wall_latency']):.1f}s, F1={np.mean(df_paired['ircot_f1']):.3f})")
    ax1.scatter(df_paired["active_wall_latency"], df_paired["active_f1"], color="#2ca02c", alpha=0.55, s=28, label=f"E-RAG Active (Med Lat={np.median(df_paired['active_wall_latency']):.1f}s, F1={np.mean(df_paired['active_f1']):.3f})")

    ax1.set_title("Latency vs Answer Quality (F1)", fontweight="bold")
    ax1.set_xlabel("Wall-Clock Response Time (seconds)")
    ax1.set_ylabel("Per-Question Token F1 Score")
    ax1.set_xscale("log")
    ax1.legend(loc="lower right")

    # Panel 2: Total Token Cost vs Gold Paragraph Recall
    ax2.scatter(df_paired["ircot_total_tokens"], df_paired["ircot_recall"], color="#ff7f0e", alpha=0.55, s=28, label=f"IRCoT (Mean Tok={np.mean(df_paired['ircot_total_tokens']):.0f}, Rec={np.mean(df_paired['ircot_recall']):.3f})")
    ax2.scatter(df_paired["active_total_tokens"], df_paired["active_recall"], color="#2ca02c", alpha=0.55, s=28, label=f"E-RAG Active (Mean Tok={np.mean(df_paired['active_total_tokens']):.0f}, Rec={np.mean(df_paired['active_recall']):.3f})")

    ax2.set_title("Computational Cost (Tokens) vs Retrieval Recall", fontweight="bold")
    ax2.set_xlabel("Total Input + Output Tokens per Question")
    ax2.set_ylabel("Gold Paragraph Recall")
    ax2.legend(loc="lower right")

    fig.suptitle("IRCoT vs E-RAG Active Resource Trade-Offs ($N=400$)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_k3_6_and_7_resource_tradeoffs")


def main():
    paired_csv = "research/methodology_audit/validated_plot_data/ircot_vs_active_paired.csv"
    out_dir = "research/methodology_audit/ircot_vs_erag_active_plots"

    print(f"=== [IRCoT vs E-RAG Active Plot Generation] Loading {paired_csv} ===")
    df_paired = pd.read_csv(paired_csv)

    print("Generating Figure K3.1 & K3.2: F1 Difference Distribution & ECDF...")
    plot_k3_1_and_2_f1_diff(df_paired, out_dir)

    print("Generating Figure K3.3: IRCoT vs Active F1 Scatter Plot...")
    plot_k3_3_f1_scatter(df_paired, out_dir)

    print("Generating Figure K3.4: Correctness Transition Breakdown...")
    plot_k3_4_correctness_transition(df_paired, out_dir)

    print("Generating Figure K3.5: Retrieval Recall Difference Histogram...")
    plot_k3_5_recall_diff(df_paired, out_dir)

    print("Generating Figure K3.6 & K3.7: Resource Trade-Offs (Latency & Tokens)...")
    plot_k3_6_and_7_tradeoffs(df_paired, out_dir)

    print("\nAll Part K3 figures successfully generated in vector PDF and 300 DPI PNG!")


if __name__ == "__main__":
    main()
