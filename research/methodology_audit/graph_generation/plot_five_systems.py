"""Generate publication-ready figures for all five systems on Test-400 (Part J).

Outputs both high-resolution PNG (300 DPI) and vector PDF into:
- research/methodology_audit/five_system_plots/
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Styling configuration for academic paper (Springer/IEEE style)
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

PALETTE = {
    "Closed-Book": "#7f7f7f",     # Neutral grey
    "Standard RAG": "#d62728",    # Carmine Red
    "IRCoT": "#ff7f0e",           # Deep Orange
    "E-RAG Observe": "#1f77b4",   # Steel Blue
    "E-RAG Active": "#2ca02c",    # Forest Green
}

SYSTEM_ORDER = ["Closed-Book", "Standard RAG", "IRCoT", "E-RAG Observe", "E-RAG Active"]


def save_fig(fig, out_dir: str, base_name: str):
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{base_name}.png")
    pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path} and {pdf_path}")


def plot_j1_main_performance(df_agg: pd.DataFrame, out_dir: str):
    """J1: Main performance comparison across systems."""
    metrics = [
        ("Exact Match", "Exact Match", [0, 0.6]),
        ("Token F1", "Token F1", [0, 0.7]),
        ("Gold Recall", "Gold Paragraph Recall", [0, 1.0]),
        ("Selective EM", "Selective EM", [0, 0.7]),
        ("Coverage", "Coverage", [0, 1.0]),
        ("All-Gold Rate", "All-Gold Rate", [0, 1.0]),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    axes = axes.flatten()

    for idx, (m_label, col_name, ylim) in enumerate(metrics):
        ax = axes[idx]
        x_pos = np.arange(len(SYSTEM_ORDER))
        vals = []
        for s in SYSTEM_ORDER:
            row = df_agg[df_agg["System"] == s]
            val = row[col_name].values[0] if len(row) > 0 else np.nan
            vals.append(val)

        bars = ax.bar(x_pos, vals, color=[PALETTE[s] for s in SYSTEM_ORDER], edgecolor="black", width=0.6, alpha=0.9)
        ax.set_title(m_label, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["Closed", "Std RAG", "IRCoT", "Observe", "Active"], rotation=25, ha="right")
        ax.set_ylim(ylim)

        for bar, val in zip(bars, vals):
            if np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, 0.05, "N/A", ha="center", va="bottom", fontsize=9, color="gray", fontstyle="italic")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, val + (ylim[1] * 0.02), f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("HotpotQA Test-400: Main Benchmark Performance Comparison ($N=400$)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_j1_main_performance_comparison")


def plot_j2_f1_distribution(df_per_q: pd.DataFrame, out_dir: str):
    """J2: Question-level F1 distribution histogram across all five systems."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0.0, 1.0, 21)  # 20 equal bins of width 0.05

    for sys_name in SYSTEM_ORDER:
        prefix = sys_name.lower().replace(" ", "_").replace("-", "_")
        f1_vals = df_per_q[f"{prefix}__f1"].values
        ax.hist(
            f1_vals,
            bins=bins,
            histtype="step",
            linewidth=2.2,
            label=f"{sys_name} (Mean = {np.mean(f1_vals):.3f})",
            color=PALETTE[sys_name],
            density=False,
        )

    ax.set_title("Question-Level Token F1 Score Distribution on Test-400 ($N=400$)", fontweight="bold")
    ax.set_xlabel("Token F1 Score (Withheld answers evaluated as 0.0)")
    ax.set_ylabel("Question Count")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="upper center", frameon=True)
    save_fig(fig, out_dir, "fig_j2_f1_distribution_histogram")


def plot_j3_f1_ecdf(df_per_q: pd.DataFrame, out_dir: str):
    """J3: Empirical Cumulative Distribution Function (ECDF) of F1 for all five systems."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for sys_name in SYSTEM_ORDER:
        prefix = sys_name.lower().replace(" ", "_").replace("-", "_")
        f1_vals = np.sort(df_per_q[f"{prefix}__f1"].values)
        y_vals = np.arange(1, len(f1_vals) + 1) / len(f1_vals)
        ax.step(f1_vals, y_vals, where="post", label=f"{sys_name} (Median={np.median(f1_vals):.3f})", color=PALETTE[sys_name], linewidth=2.2)

    ax.set_title("Empirical Cumulative Distribution Function (ECDF) of F1 Scores ($N=400$)", fontweight="bold")
    ax.set_xlabel("Per-Question Token F1 Score")
    ax.set_ylabel(r"Proportion of Questions (F1 $\leq$ x)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)
    save_fig(fig, out_dir, "fig_j3_f1_cumulative_ecdf")


def plot_j4_confidence_and_calibration(df_per_q: pd.DataFrame, out_dir: str):
    """J4: Confidence histogram and reliability diagram for calibrated systems."""
    # Subplot 1: Confidence Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for sys_name in ["E-RAG Observe", "E-RAG Active"]:
        prefix = sys_name.lower().replace(" ", "_").replace("-", "_")
        confs = df_per_q[f"{prefix}__answer_conf"].dropna().values
        ax1.hist(confs, bins=np.linspace(0.0, 1.0, 21), histtype="step", linewidth=2.2, label=f"{sys_name} ($N={len(confs)}$)", color=PALETTE[sys_name])

    ax1.set_title("Predicted Epistemic Confidence Distribution", fontweight="bold")
    ax1.set_xlabel(r"Effective Confidence $\sigma(z_{\mathrm{ans}})$")
    ax1.set_ylabel("Question Count")
    ax1.set_xlim(-0.02, 1.02)
    ax1.legend(loc="upper left")

    # Subplot 2: Reliability Diagram (Active vs Perfect)
    ax2.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.8)
    for sys_name in ["E-RAG Active", "E-RAG Observe"]:
        prefix = sys_name.lower().replace(" ", "_").replace("-", "_")
        confs = df_per_q[f"{prefix}__answer_conf"].values
        ems = df_per_q[f"{prefix}__em"].values
        
        bins = np.linspace(0.0, 1.0, 11)
        bin_confs, bin_accs = [], []
        for i in range(10):
            m = (confs >= bins[i]) & (confs <= bins[i+1] if i==9 else confs < bins[i+1])
            if np.any(m):
                bin_confs.append(np.mean(confs[m]))
                bin_accs.append(np.mean(ems[m]))
        
        ax2.plot(bin_confs, bin_accs, "o-", label=f"{sys_name}", color=PALETTE[sys_name], markersize=6)

    ax2.set_title("Reliability Diagram (Calibration Curve, 10 Bins)", fontweight="bold")
    ax2.set_xlabel("Mean Predicted Confidence")
    ax2.set_ylabel("Empirical Exact Match Accuracy")
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper left")

    fig.suptitle("Epistemic Uncertainty Calibration on Test-400 (Note: Standard RAG, IRCoT, Closed-Book N/A)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_j4_confidence_and_calibration")


def plot_j5_retrieval_recall(df_per_q: pd.DataFrame, out_dir: str):
    """J5: Retrieval recall histogram and ECDF for retrieval-enabled systems."""
    retrieval_systems = ["Standard RAG", "IRCoT", "E-RAG Observe", "E-RAG Active"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Histogram
    bins = np.linspace(0.0, 1.0, 11)
    for sys_name in retrieval_systems:
        prefix = sys_name.lower().replace(" ", "_").replace("-", "_")
        recs = df_per_q[f"{prefix}__recall"].dropna().values
        ax1.hist(recs, bins=bins, histtype="step", linewidth=2.2, label=f"{sys_name} (Mean={np.mean(recs):.3f})", color=PALETTE[sys_name])

    ax1.set_title("Gold Paragraph Recall Distribution ($N=400$)", fontweight="bold")
    ax1.set_xlabel("Gold Paragraph Recall")
    ax1.set_ylabel("Question Count")
    ax1.set_xlim(-0.02, 1.02)
    ax1.legend(loc="upper left")

    # Panel 2: ECDF
    for sys_name in retrieval_systems:
        prefix = sys_name.lower().replace(" ", "_").replace("-", "_")
        recs = np.sort(df_per_q[f"{prefix}__recall"].dropna().values)
        y = np.arange(1, len(recs) + 1) / len(recs)
        ax2.step(recs, y, where="post", label=f"{sys_name} (All-Gold={np.mean(recs == 1.0):.1%})", color=PALETTE[sys_name], linewidth=2.2)

    ax2.set_title("Empirical Cumulative Retrieval Recall ECDF", fontweight="bold")
    ax2.set_xlabel("Gold Paragraph Recall")
    ax2.set_ylabel(r"Proportion of Questions (Recall $\leq$ x)")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(loc="upper left")

    fig.suptitle("Retrieval Quality Across Systems (Closed-Book Excluded as Non-Retrieval)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_j5_retrieval_recall_analysis")


def plot_j6_latency_and_efficiency(df_per_q: pd.DataFrame, df_agg: pd.DataFrame, out_dir: str):
    """J6: Wall latency distribution boxplots and token/call consumption."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Boxplot of Wall Latency
    latency_data = []
    for s in SYSTEM_ORDER:
        prefix = s.lower().replace(" ", "_").replace("-", "_")
        latency_data.append(df_per_q[f"{prefix}__wall_latency"].values)

    box = ax1.boxplot(latency_data, patch_artist=True, tick_labels=["Closed", "Std RAG", "IRCoT", "Observe", "Active"], showmeans=True)
    for patch, s in zip(box["boxes"], SYSTEM_ORDER):
        patch.set_facecolor(PALETTE[s])
        patch.set_alpha(0.7)

    ax1.set_title("End-to-End Wall Latency Distribution per Question", fontweight="bold")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    # Panel 2: Calls & Token Resource Comparison
    x_pos = np.arange(len(SYSTEM_ORDER))
    llm_calls = [df_agg[df_agg["System"] == s]["Mean LLM Calls/Q"].values[0] for s in SYSTEM_ORDER]
    tokens = [df_agg[df_agg["System"] == s]["Mean Total Tokens"].values[0] for s in SYSTEM_ORDER]

    ax2_twin = ax2.twinx()
    b1 = ax2.bar(x_pos - 0.15, llm_calls, width=0.3, color="#1f77b4", label="Mean LLM Calls/Q", alpha=0.85)
    b2 = ax2_twin.bar(x_pos + 0.15, tokens, width=0.3, color="#ff7f0e", label="Mean Tokens/Q", alpha=0.85)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["Closed", "Std RAG", "IRCoT", "Observe", "Active"], rotation=25, ha="right")
    ax2.set_ylabel("LLM Invocations per Question", color="#1f77b4", fontweight="bold")
    ax2_twin.set_ylabel("Total Tokens per Question", color="#ff7f0e", fontweight="bold")
    ax2.set_title("Computational Overhead Across Systems", fontweight="bold")

    fig.suptitle("Computational Efficiency & Latency Trade-Off ($N=400$)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir, "fig_j6_latency_and_efficiency")


def main():
    validated_csv = "research/methodology_audit/validated_plot_data/test400_five_systems_per_question.csv"
    agg_csv = "research/methodology_audit/validated_plot_data/test400_aggregate_metrics.csv"
    out_dir = "research/methodology_audit/five_system_plots"

    print(f"=== [Five-System Plot Generation] Loading {validated_csv} ===")
    df_per_q = pd.read_csv(validated_csv)
    df_agg = pd.read_csv(agg_csv)

    print("Generating Figure J1: Main Performance Comparison...")
    plot_j1_main_performance(df_agg, out_dir)

    print("Generating Figure J2: Token F1 Score Distribution Histogram...")
    plot_j2_f1_distribution(df_per_q, out_dir)

    print("Generating Figure J3: Token F1 Cumulative ECDF...")
    plot_j3_f1_ecdf(df_per_q, out_dir)

    print("Generating Figure J4: Epistemic Confidence & Reliability Calibration...")
    plot_j4_confidence_and_calibration(df_per_q, out_dir)

    print("Generating Figure J5: Retrieval Recall Histogram & ECDF...")
    plot_j5_retrieval_recall(df_per_q, out_dir)

    print("Generating Figure J6: Latency & Computational Efficiency...")
    plot_j6_latency_and_efficiency(df_per_q, df_agg, out_dir)

    print("\nAll Part J figures successfully generated in vector PDF and 300 DPI PNG!")


if __name__ == "__main__":
    main()
