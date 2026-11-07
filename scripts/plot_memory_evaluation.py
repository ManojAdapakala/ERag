"""Generate publication-ready evaluation plots for Verified-Belief Episodic Memory (Experiment E5).

Evaluates:
1. Memory ON vs. Memory OFF Performance (Accuracy, F1, Retrievals).
2. Question-by-Question Stream Dynamics (Q1 to Q30 memory reuse events).
3. Planted Wrong-Belief Override Dynamics (Purging contradicted beliefs).
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


def main():
    res_dir = "research/erag_vs_ircot_study"
    artifact_dir = "C:/Users/hp/.gemini/antigravity/brain/f1e5200d-a581-4031-af73-9be369bc1639"
    os.makedirs(res_dir, exist_ok=True)

    nomem = [json.loads(l) for l in open("results/stream_nomem.jsonl", encoding="utf-8") if l.strip()]
    mem = [json.loads(l) for l in open("results/stream_mem.jsonl", encoding="utf-8") if l.strip()]
    planted = [json.loads(l) for l in open("results/stream_planted.jsonl", encoding="utf-8") if l.strip()]
    plant_defs = json.load(open("data/manifests/planted.json", encoding="utf-8"))

    n = len(mem)
    nomem_em = np.mean([r["em"] for r in nomem]) * 100
    mem_em = np.mean([r["em"] for r in mem]) * 100
    nomem_f1 = np.mean([r["f1"] for r in nomem]) * 100
    mem_f1 = np.mean([r["f1"] for r in mem]) * 100

    nomem_ret = np.mean([r["retrievals"] for r in nomem])
    mem_ret = np.mean([r["retrievals"] for r in mem])

    reused = [i + 1 for i, r in enumerate(mem) if r.get("mem_used", 0) > 0]

    # Identify overridden planted beliefs
    all_overrides = []
    for r in planted:
        for ev in r.get("mem_events", []):
            if ev.get("event") == "overridden":
                all_overrides.append(ev.get("belief", ""))
    overridden_set = set(all_overrides)

    # Create 3-panel figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    # -------------------------------------------------------------
    # Panel 1: Aggregate Benchmark Metrics (Memory ON vs OFF)
    # -------------------------------------------------------------
    categories = ["Exact Match (%)", "Token F1 (%)", "Retrievals / Q"]
    nomem_vals = [nomem_em, nomem_f1, nomem_ret]
    mem_vals = [mem_em, mem_f1, mem_ret]

    x = np.arange(len(categories))
    w = 0.35

    b1 = ax1.bar(x - w/2, nomem_vals, w, label="E-RAG (Memory OFF)", color="#7f7f7f", edgecolor="black", linewidth=0.8)
    b2 = ax1.bar(x + w/2, mem_vals, w, label="E-RAG (Memory ON)", color="#1f77b4", edgecolor="black", linewidth=0.8)

    ax1.set_ylabel("Score / Count", fontsize=11, fontweight="bold")
    ax1.set_title("A. Memory Impact on Clustered Stream (N = 30)", fontsize=11.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 60)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax1.legend(loc="upper right", fontsize=9.5)

    for r in b1:
        h = r.get_height()
        val_str = f"{h:.1f}%" if h > 5 else f"{h:.2f}"
        ax1.annotate(val_str, xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    for r in b2:
        h = r.get_height()
        val_str = f"{h:.1f}%" if h > 5 else f"{h:.2f}"
        ax1.annotate(val_str, xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 2: Question-by-Question Stream Timeline
    # -------------------------------------------------------------
    qs = np.arange(1, n + 1)
    ret_counts = [r["retrievals"] for r in mem]
    colors = ["#2ca02c" if q in reused else "#aec7e8" for q in qs]

    bars = ax2.bar(qs, ret_counts, color=colors, edgecolor="black", linewidth=0.5, width=0.7)
    ax2.set_xlabel("Question Sequence Index in Stream (1 to 30)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("External Retrievals Executed", fontsize=10.5, fontweight="bold")
    ax2.set_title("B. Stream Execution & Belief Reuse Events", fontsize=11.5, fontweight="bold")
    ax2.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax2.set_ylim(0, 4)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")

    # Callouts for reuse events
    for q in reused:
        ax2.annotate(
            f"Belief Reused\n(Q{q})",
            xy=(q, ret_counts[q-1]),
            xytext=(q - 2, ret_counts[q-1] + 0.5),
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5),
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#eef9ee", ec="#2ca02c", lw=1),
        )

    # Dummy bars for legend
    ax2.bar([0], [0], color="#aec7e8", label="Fresh Retrieval Pool")
    ax2.bar([0], [0], color="#2ca02c", label=f"Memory Reuse Event (n={len(reused)})")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xlim(0, 31)

    # -------------------------------------------------------------
    # Panel 3: Planted Synthetic Wrong-Belief Override Dynamics
    # -------------------------------------------------------------
    planted_short = [
        "Wolfgang Becker (French director)",
        "Eileen Heckart (Grammy winner)",
        "Bae Doona (Tokyo birthplace)",
        "Stanley Kubrick (UK citizen)",
        "Gattaca (1995 release date)",
    ]
    status_colors = ["#2ca02c" if p in overridden_set else "#d62728" for p in [b["text"] for b in plant_defs]]
    labels = ["Purged (Contradiction Override)" if p in overridden_set else "Dormant (Un-queried in stream)" for p in [b["text"] for b in plant_defs]]

    yp = np.arange(len(planted_short))
    ax3.barh(yp, [1.0] * len(planted_short), color=status_colors, edgecolor="black", height=0.55)
    ax3.set_yticks(yp)
    ax3.set_yticklabels(planted_short, fontsize=9.5, fontweight="bold")
    ax3.set_xlabel("Override Status", fontsize=10.5, fontweight="bold")
    ax3.set_title("C. Planted Contradiction Override Test (Exp E5)", fontsize=11.5, fontweight="bold")
    ax3.set_xlim(0, 1.25)
    ax3.set_xticks([])

    for i, lbl in enumerate(labels):
        txt_color = "white" if "Purged" in lbl else "#333333"
        bg_color = "#1b661b" if "Purged" in lbl else "#fbebeb"
        ax3.annotate(lbl, xy=(0.05, yp[i]), va="center", fontsize=8.5, fontweight="bold", color=txt_color, bbox=dict(boxstyle="round,pad=0.2", fc=bg_color, ec="none"))

    plt.tight_layout()
    p_out = os.path.join(res_dir, "fig_memory_stream_evaluation.png")
    plt.savefig(p_out)
    plt.savefig(os.path.join(artifact_dir, "fig_memory_stream_evaluation.png"))
    plt.close()

    print(f"[+] Saved memory evaluation figure to: {p_out}")


if __name__ == "__main__":
    main()

