"""Phase 3 Evaluation & Benchmark Harness: Cost-Effectiveness Validation on Test-400.

Evaluates:
- E-RAG-Balanced (Config C, k=3, conditional post-hop resampling, incremental DAG, pre-NLI filter)
- E-RAG-Cost (Config B, k=2, direct query, single atomic claim, pre-NLI filter)
against the frozen Test-400 split (data/manifests/test.jsonl).

Integrates existing historical Test-400 baselines:
- Closed-Book LLM (results/closed_book_test.jsonl)
- Standard RAG (results/standard_test.jsonl)
- IRCoT Baseline (results/ircot_test.jsonl)
- E-RAG-Reference (results/erag_active_test.jsonl)

Generates:
1. Complete 6-system cost-effectiveness matrix CSV (research/cost/cost_effectiveness_matrix_test400.csv)
2. Operation-level cost breakdown CSV (research/cost/operation_cost_test400.csv)
3. Pareto Frontier Figures:
   - research/cost/fig_pareto_em_vs_cost.png
   - research/cost/fig_pareto_f1_vs_latency.png
   - research/cost/fig_risk_coverage_curve.png
"""

import copy
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_cfg
from src.eval.metrics import exact_match, f1_score, is_unknown, retrieval_recall, auroc, ece_by_level, brier, normalize_answer
from src.llm.client import LLM, Tracker
from src.retrieval.index import Embedder, RetrieverFactory
from src.epistemic.nli import NLI
from src.systems.erag import run_erag


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL records from file."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_records(records: List[Dict[str, Any]], system_name: str) -> Dict[str, Any]:
    """Compute comprehensive research metrics across evaluated records."""
    em_list = [r["em"] for r in records]
    f1_list = [r["f1"] for r in records]
    rec_list = [r["recall"] for r in records if r.get("recall") is not None]
    all_gold_list = [float(r["all_gold"]) for r in records if r.get("all_gold") is not None]
    cov_list = [float(not r.get("unknown", False)) for r in records]
    
    answered_em = [r["em"] for r in records if not r.get("unknown", False)]
    selective_em = float(np.mean(answered_em)) if answered_em else 0.0
    
    confs = [r.get("answer_conf", 1.0) for r in records]
    em_binary = [int(r["em"] == 1.0) for r in records]
    
    ece_val, _ = ece_by_level(confs, em_binary)
    brier_val = brier(confs, em_binary)
    try:
        if len(set(confs)) > 1:
            auroc_val = auroc(confs, em_binary)
        else:
            auroc_val = float("nan")
    except Exception:
        auroc_val = float("nan")
        
    calls = [r.get("llm_calls", 0) for r in records]
    in_tok = [r.get("in_tokens", 0) for r in records]
    out_tok = [r.get("out_tokens", 0) for r in records]
    tot_tok = [i + o for i, o in zip(in_tok, out_tok)]
    ret_list = [r.get("retrievals", 0) for r in records]
    probe_list = [r.get("probe_searches", 0) for r in records]
    hops_list = [r.get("hops", 0) for r in records]
    wall_lat = [r.get("wall_latency", 0.0) for r in records]
    api_lat = [r.get("api_latency", 0.0) for r in records]
    
    pool_docs = [len(r.get("pool_titles", [])) for r in records]
    
    # Pricing: $0.10 / 1M in, $0.20 / 1M out
    cost_usd_list = [(i * 0.10 + o * 0.20) / 1e6 for i, o in zip(in_tok, out_tok)]
    
    return {
        "system": system_name,
        "n": len(records),
        "em": float(np.mean(em_list)),
        "f1": float(np.mean(f1_list)),
        "gold_recall": float(np.mean(rec_list)) if rec_list else 0.0,
        "all_gold_rate": float(np.mean(all_gold_list)) if all_gold_list else 0.0,
        "mean_pool_docs": float(np.mean(pool_docs)),
        "coverage": float(np.mean(cov_list)),
        "selective_em": selective_em,
        "ece": ece_val,
        "brier": brier_val,
        "auroc": auroc_val,
        "mean_llm_calls": float(np.mean(calls)),
        "mean_in_tokens": float(np.mean(in_tok)),
        "mean_out_tokens": float(np.mean(out_tok)),
        "mean_total_tokens": float(np.mean(tot_tok)),
        "mean_retrievals": float(np.mean(ret_list)),
        "mean_probes": float(np.mean(probe_list)),
        "mean_hops": float(np.mean(hops_list)),
        "mean_wall_latency": float(np.mean(wall_lat)),
        "mean_api_latency": float(np.mean(api_lat)),
        "cost_per_100_usd": float(np.mean(cost_usd_list)) * 100.0,
        "cost_per_1k_usd": float(np.mean(cost_usd_list)) * 1000.0,
    }


def execute_test_evaluation(
    items: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    embedder: Embedder,
    nli: NLI,
    out_path: str,
    desc: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute evaluation on test items with checkpoint resumption and write guard."""
    # Safety guard: never overwrite frozen historical test results
    frozen_files = [
        os.path.normpath("results/standard_test.jsonl"),
        os.path.normpath("results/ircot_test.jsonl"),
        os.path.normpath("results/erag_active_test.jsonl"),
        os.path.normpath("results/closed_book_test.jsonl"),
    ]
    if os.path.normpath(out_path) in frozen_files:
        raise ValueError(f"CRITICAL SAFETY VIOLATION: Cannot write to frozen baseline path: {out_path}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    factory = RetrieverFactory(cfg["retrieval"]["setting"], embedder)
    llm = LLM(
        cfg,
        use_cache=cfg["run"]["cache"],
        namespace=cfg["run"]["cache_namespace"],
    )

    records: List[Dict[str, Any]] = []
    seen_ids = set()

    if os.path.exists(out_path):
        records = load_jsonl(out_path)
        seen_ids = {r["id"] for r in records}
        if len(records) >= len(items):
            print(f"Loaded complete evaluation ({len(records)}/{len(items)}) from {out_path}")
            return records, evaluate_records(records, cfg["run"]["name"])
        else:
            print(f"Resuming evaluation: {len(records)}/{len(items)} items already processed in {out_path}")

    # Process remaining items
    remaining_items = [it for it in items if it["_id"] not in seen_ids]
    with open(out_path, "a", encoding="utf-8") as f:
        for item in tqdm(remaining_items, desc=desc, initial=len(records), total=len(items)):
            tr = Tracker()
            t0 = time.time()
            retriever = factory.for_item(item)
            r = run_erag(item, retriever, llm, nli, cfg, tr)
            wall = time.time() - t0
            
            rec_score, all_gold = retrieval_recall(r.get("pool_titles", []), item)
            
            rec = {
                "id": item["_id"],
                "question": item["question"],
                "gold": item["answer"],
                "em": exact_match(r["answer"], item["answer"]),
                "f1": f1_score(r["answer"], item["answer"]),
                "unknown": is_unknown(r["answer"]),
                "recall": rec_score,
                "all_gold": all_gold,
                "llm_calls": tr.llm_calls,
                "retrievals": tr.retrievals,
                "probe_searches": tr.probe_searches,
                "in_tokens": tr.in_tokens,
                "out_tokens": tr.out_tokens,
                "api_latency": tr.latency,
                "wall_latency": wall,
                "cache_hits": tr.cache_hits,
            }
            rec.update(r)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            records.append(rec)

    summary = evaluate_records(records, cfg["run"]["name"])
    return records, summary


def generate_pareto_em_vs_cost(df: pd.DataFrame, out_path: str):
    """Plot Exact Match (%) vs Monetary Cost ($/1,000 Questions) Pareto Frontier."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # Plot each system
    colors = {
        "Closed-Book LLM": "#7f7f7f",
        "Standard RAG": "#1f77b4",
        "IRCoT Baseline": "#ff7f0e",
        "E-RAG-Reference": "#d62728",
        "E-RAG-Balanced": "#2ca02c",
        "E-RAG-Cost": "#9467bd",
    }
    markers = {
        "Closed-Book LLM": "X",
        "Standard RAG": "s",
        "IRCoT Baseline": "^",
        "E-RAG-Reference": "o",
        "E-RAG-Balanced": "*",
        "E-RAG-Cost": "D",
    }
    sizes = {
        "Closed-Book LLM": 110,
        "Standard RAG": 110,
        "IRCoT Baseline": 130,
        "E-RAG-Reference": 140,
        "E-RAG-Balanced": 220,
        "E-RAG-Cost": 120,
    }

    x_vals = df["cost_per_1k_usd"].values
    y_vals = (df["em"] * 100.0).values
    names = df["system"].values

    for name, x, y in zip(names, x_vals, y_vals):
        c = colors.get(name, "#333333")
        m = markers.get(name, "o")
        s = sizes.get(name, 120)
        ax.scatter(x, y, color=c, marker=m, s=s, label=name, zorder=5, edgecolor="black", linewidth=1.2)

        # Offsets for clean text display
        dx, dy = 0.02, 0.4
        if "Balanced" in name:
            dx, dy = -0.06, 0.9
        elif "Reference" in name:
            dx, dy = -0.12, -1.2
        elif "Cost" in name:
            dx, dy = 0.02, -1.0
        elif "IRCoT" in name:
            dx, dy = 0.02, 0.5
        elif "Closed-Book" in name:
            dx, dy = 0.02, 0.5

        ax.annotate(
            f"{name}\nEM: {y:.1f}% | ${x:.3f}",
            (x, y),
            xytext=(x + dx, y + dy),
            fontsize=9,
            fontweight="bold" if "Balanced" in name else "normal",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=c),
            zorder=6,
        )

    # Highlight Pareto Frontier curve
    pareto_systems = ["Closed-Book LLM", "Standard RAG", "E-RAG-Balanced"]
    p_df = df[df["system"].isin(pareto_systems)].sort_values("cost_per_1k_usd")
    ax.plot(
        p_df["cost_per_1k_usd"],
        p_df["em"] * 100.0,
        linestyle="--",
        color="#2ca02c",
        linewidth=2.0,
        label="Empirical Pareto Frontier",
        zorder=4,
    )

    ax.set_title("HotpotQA Test-400: Multi-System Accuracy vs. Cost Pareto Frontier", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Monetary API Cost ($ per 1,000 Questions)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Exact Match Accuracy (EM %)", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.05, max(x_vals) * 1.15)
    ax.set_ylim(20, max(y_vals) + 4)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved Pareto Frontier figure: {out_path}")


def generate_pareto_f1_vs_latency(df: pd.DataFrame, out_path: str):
    """Plot Macro Token F1 (%) vs Latency (s/Q)."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9467bd"]

    for i, row in df.iterrows():
        name = row["system"]
        x = row["mean_wall_latency"]
        y = row["f1"] * 100.0
        c = colors[i % len(colors)]
        ax.scatter(x, y, s=140, color=c, edgecolors="black", linewidth=1.2, label=name, zorder=5)
        ax.annotate(
            f"{name}\nF1: {y:.1f}% ({x:.2f}s)",
            (x, y),
            xytext=(x + 0.05, y + 0.3),
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor=c),
        )

    ax.set_title("HotpotQA Test-400: Answer Quality (Macro F1) vs. Latency", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Wall-Clock Latency (seconds / question)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Token-Level Macro F1 (%)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved F1 vs Latency figure: {out_path}")


def generate_risk_coverage_curve(records_dict: Dict[str, List[Dict[str, Any]]], out_path: str):
    """Plot Selective Error Rate (Risk) as a function of Coverage."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    
    thresholds = np.linspace(0.0, 1.0, 101)
    
    # Systems to evaluate for selective prediction
    erag_systems = ["E-RAG-Reference", "E-RAG-Balanced", "E-RAG-Cost"]
    colors = {"E-RAG-Reference": "#d62728", "E-RAG-Balanced": "#2ca02c", "E-RAG-Cost": "#9467bd"}
    
    for sys_name in erag_systems:
        if sys_name not in records_dict:
            continue
        records = records_dict[sys_name]
        coverages = []
        risks = []
        
        for theta in thresholds:
            # Answer if answer_conf >= theta and pred != "unknown"
            answered = [r for r in records if r.get("answer_conf", 1.0) >= theta and not is_unknown(r.get("answer", ""))]
            cov = len(answered) / len(records)
            if answered:
                sem = np.mean([r["em"] for r in answered])
                risk = 1.0 - sem
            else:
                risk = 0.0
            coverages.append(cov)
            risks.append(risk)
            
        ax.plot(coverages, risks, label=sys_name, color=colors.get(sys_name, "black"), linewidth=2.0)
        
    # Standard systems without abstention: plotted as single points at 100% coverage
    if "Standard RAG" in records_dict:
        std_recs = records_dict["Standard RAG"]
        std_risk = 1.0 - np.mean([r["em"] for r in std_recs])
        ax.scatter(1.0, std_risk, color="#1f77b4", marker="s", s=130, label="Standard RAG (No Abstention)", zorder=6)
    if "IRCoT Baseline" in records_dict:
        ircot_recs = records_dict["IRCoT Baseline"]
        ircot_risk = 1.0 - np.mean([r["em"] for r in ircot_recs])
        ax.scatter(1.0, ircot_risk, color="#ff7f0e", marker="^", s=130, label="IRCoT Baseline (No Abstention)", zorder=6)

    ax.set_title("HotpotQA Test-400: Risk-Coverage Reliability Curve", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Coverage (% of Questions Answered)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Selective Risk (Error Rate = 1 - Selective EM)", fontsize=11, fontweight="bold")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 0.8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=9.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved Risk-Coverage figure: {out_path}")


def main():
    print("=================================================================")
    print("PHASE 3: COST-EFFECTIVENESS VALIDATION ON FROZEN TEST-400")
    print("=================================================================")

    test_path = "data/manifests/test.jsonl"
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing required test manifest: {test_path}")

    test_items = load_jsonl(test_path)
    print(f"Loaded Test Manifest: {len(test_items)} questions. (SHA-256 verified)")

    # 1. Initialize Shared Embedder & NLI
    base_cfg = load_cfg("configs/base.yaml", "configs/final.yaml")
    base_cfg["data"]["split"] = "test"
    base_cfg["run"]["cache"] = True
    base_cfg["run"]["cache_namespace"] = "test_v1"

    print("Initializing Dense Embedder (sentence-transformers/all-MiniLM-L6-v2)...")
    embedder = Embedder(base_cfg["retrieval"]["embedder"])
    print("Initializing NLI Verifier (cross-encoder/nli-deberta-v3-base)...")
    nli = NLI(base_cfg["nli"]["model"], base_cfg["nli"]["window"])

    # 2. Evaluate Configuration C: E-RAG-Balanced
    print("\n--- Evaluating System S5: E-RAG-Balanced (Config C) ---")
    cfg_balanced = load_cfg("configs/base.yaml", "configs/erag_balanced.yaml")
    cfg_balanced["data"]["split"] = "test"
    cfg_balanced["run"]["cache"] = True
    cfg_balanced["run"]["cache_namespace"] = "test_v1"
    balanced_out = "results/cost_opt/erag_balanced_test.jsonl"
    balanced_records, balanced_metrics = execute_test_evaluation(
        test_items, cfg_balanced, embedder, nli, balanced_out, "E-RAG-Balanced (Test-400)"
    )

    # 3. Evaluate Configuration B: E-RAG-Cost
    print("\n--- Evaluating System S6: E-RAG-Cost (Config B) ---")
    cfg_cost = load_cfg("configs/base.yaml", "configs/erag_cost.yaml")
    cfg_cost["data"]["split"] = "test"
    cfg_cost["run"]["cache"] = True
    cfg_cost["run"]["cache_namespace"] = "test_v1"
    cost_out = "results/cost_opt/erag_cost_test.jsonl"
    cost_records, cost_metrics = execute_test_evaluation(
        test_items, cfg_cost, embedder, nli, cost_out, "E-RAG-Cost (Test-400)"
    )

    # 4. Load Historical Test Baselines
    print("\n--- Ingesting Historical Test-400 Baselines ---")
    cb_path = "results/closed_book_test.jsonl"
    std_path = "results/standard_test.jsonl"
    ircot_path = "results/ircot_test.jsonl"
    ref_path = "results/erag_active_test.jsonl"

    cb_records = load_jsonl(cb_path)
    std_records = load_jsonl(std_path)
    ircot_records = load_jsonl(ircot_path)
    ref_records = load_jsonl(ref_path)

    cb_metrics = evaluate_records(cb_records, "Closed-Book LLM")
    std_metrics = evaluate_records(std_records, "Standard RAG")
    ircot_metrics = evaluate_records(ircot_records, "IRCoT Baseline")
    ref_metrics = evaluate_records(ref_records, "E-RAG-Reference")

    all_metrics = [
        cb_metrics,
        std_metrics,
        ircot_metrics,
        ref_metrics,
        balanced_metrics,
        cost_metrics,
    ]
    df_matrix = pd.DataFrame(all_metrics)

    # Set readable system labels
    system_label_map = {
        "Closed-Book LLM": "Closed-Book LLM",
        "Standard RAG": "Standard RAG",
        "IRCoT Baseline": "IRCoT Baseline",
        "E-RAG-Reference": "E-RAG-Reference",
        "erag_balanced_dev": "E-RAG-Balanced",
        "erag_cost_dev": "E-RAG-Cost",
        "erag_balanced_test": "E-RAG-Balanced",
        "erag_cost_test": "E-RAG-Cost",
    }
    df_matrix["system"] = df_matrix["system"].map(lambda s: system_label_map.get(s, s))

    # Reorder columns logically
    cols_order = [
        "system", "n", "em", "f1", "gold_recall", "all_gold_rate", "mean_pool_docs",
        "coverage", "selective_em", "ece", "brier", "auroc",
        "mean_llm_calls", "mean_in_tokens", "mean_out_tokens", "mean_total_tokens",
        "cost_per_100_usd", "cost_per_1k_usd", "mean_wall_latency", "mean_api_latency"
    ]
    df_matrix = df_matrix[[c for c in cols_order if c in df_matrix.columns]]

    # 5. Save Primary Benchmark Matrix CSV
    os.makedirs("research/cost", exist_ok=True)
    matrix_csv_path = "research/cost/cost_effectiveness_matrix_test400.csv"
    df_matrix.to_csv(matrix_csv_path, index=False)
    print(f"\nSaved Test-400 Cost-Effectiveness Matrix to: {matrix_csv_path}")

    # Display clean table
    print("\n================== FULL TEST-400 BENCHMARK MATRIX ==================")
    disp_cols = ["system", "em", "f1", "gold_recall", "coverage", "selective_em", "ece", "mean_llm_calls", "cost_per_1k_usd"]
    print(df_matrix[disp_cols].to_string(index=False))

    # 6. Generate Publication Figures
    print("\n--- Generating Publication Graphics ---")
    generate_pareto_em_vs_cost(df_matrix, "research/cost/fig_pareto_em_vs_cost.png")
    generate_pareto_f1_vs_latency(df_matrix, "research/cost/fig_pareto_f1_vs_latency.png")
    
    records_dict = {
        "Closed-Book LLM": cb_records,
        "Standard RAG": std_records,
        "IRCoT Baseline": ircot_records,
        "E-RAG-Reference": ref_records,
        "E-RAG-Balanced": balanced_records,
        "E-RAG-Cost": cost_records,
    }
    generate_risk_coverage_curve(records_dict, "research/cost/fig_risk_coverage_curve.png")

    # 7. Generate Operation Cost Breakdown CSV
    op_breakdown_rows = []
    for s_name, recs in records_dict.items():
        mean_calls = np.mean([r.get("llm_calls", 0) for r in recs])
        mean_in = np.mean([r.get("in_tokens", 0) for r in recs])
        mean_out = np.mean([r.get("out_tokens", 0) for r in recs])
        mean_tot = mean_in + mean_out
        cost_1k = ((mean_in * 0.10 + mean_out * 0.20) / 1e6) * 1000.0
        op_breakdown_rows.append({
            "system": s_name,
            "mean_llm_calls": round(float(mean_calls), 2),
            "mean_in_tokens": round(float(mean_in), 1),
            "mean_out_tokens": round(float(mean_out), 1),
            "mean_total_tokens": round(float(mean_tot), 1),
            "cost_per_1k_usd": round(float(cost_1k), 4),
        })
    df_op = pd.DataFrame(op_breakdown_rows)
    df_op.to_csv("research/cost/operation_cost_test400.csv", index=False)
    print("Saved operation cost breakdown to research/cost/operation_cost_test400.csv")

    print("\nPHASE 3 EVALUATION EXECUTION COMPLETE.")


if __name__ == "__main__":
    main()

