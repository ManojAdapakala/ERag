"""Offline Metric Enrichment Harness: Calculates Hop-by-Hop Recall, Precision, MRR, 
and Bridge Entity Resolution across all 12 benchmark conditions on HotpotQA Test-400.

Reads existing raw predictions, computes deep retrieval dynamics with ZERO LLM calls,
and outputs enriched JSONL files into `results/enriched/` and a summary CSV into
`research/retrievers/enriched_multi_hop_dynamics.csv`.
"""

import json
import os
import sys
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

MANIFEST_PATH = "data/manifests/test.jsonl"
OUT_DIR = "results/enriched"
SUMMARY_CSV = "research/retrievers/enriched_multi_hop_dynamics.csv"

CONDITIONS = [
    # Paradigm, Retriever Label, Source File, Target File
    ("Standard RAG", "BM25 (Sparse)", "results/bm25/standard_bm25_test.jsonl", "standard_bm25_test.jsonl"),
    ("Standard RAG", "MiniLM (Dense 384d)", "results/standard_test.jsonl", "standard_minilm_test.jsonl"),
    ("Standard RAG", "E5-base-v2 (Dense 768d)", "results/phase5/standard_e5_test.jsonl", "standard_e5_test.jsonl"),
    ("Standard RAG", "BGE-base-en-v1.5 (Dense 768d)", "results/phase5/standard_bge_test.jsonl", "standard_bge_test.jsonl"),
    
    ("IRCoT-Controlled", "BM25 (Sparse)", "results/bm25/ircot_bm25_test.jsonl", "ircot_bm25_test.jsonl"),
    ("IRCoT-Controlled", "MiniLM (Dense 384d)", "results/ircot_test.jsonl", "ircot_minilm_test.jsonl"),
    ("IRCoT-Controlled", "E5-base-v2 (Dense 768d)", "results/phase5/ircot_e5_test.jsonl", "ircot_e5_test.jsonl"),
    ("IRCoT-Controlled", "BGE-base-en-v1.5 (Dense 768d)", "results/phase5/ircot_bge_test.jsonl", "ircot_bge_test.jsonl"),
    
    ("E-RAG-Balanced", "BM25 (Sparse)", "results/bm25/erag_balanced_bm25_test.jsonl", "erag_balanced_bm25_test.jsonl"),
    ("E-RAG-Balanced", "MiniLM (Dense 384d)", "results/cost_opt/erag_balanced_test.jsonl", "erag_balanced_minilm_test.jsonl"),
    ("E-RAG-Balanced", "E5-base-v2 (Dense 768d)", "results/phase5/erag_balanced_e5_test.jsonl", "erag_balanced_e5_test.jsonl"),
    ("E-RAG-Balanced", "BGE-base-en-v1.5 (Dense 768d)", "results/phase5/erag_balanced_bge_test.jsonl", "erag_balanced_bge_test.jsonl"),
]


def load_manifest(path: str) -> Dict[str, Set[str]]:
    """Map question ID to set of gold paragraph titles."""
    manifest = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            q_id = item["_id"]
            gold_titles = {t for t, _ in item.get("supporting_facts", [])}
            manifest[q_id] = gold_titles
    return manifest


def compute_first_gold_rank(pool_titles: List[str], gold_titles: Set[str]) -> Tuple[int, float]:
    """Find 1-based rank of the first gold document in pool, and reciprocal rank."""
    for idx, title in enumerate(pool_titles, start=1):
        if title in gold_titles:
            return idx, 1.0 / idx
    return 0, 0.0


def enrich_record(
    record: Dict[str, Any],
    gold_titles: Set[str],
    paradigm: str
) -> Dict[str, Any]:
    """Calculate and attach new precision, recall progression, and MRR metrics."""
    pool_titles = record.get("pool_titles", [])
    gold_in_pool = set(pool_titles) & gold_titles
    n_gold = max(len(gold_titles), 1)
    n_pool = max(len(pool_titles), 1)

    precision = len(gold_in_pool) / n_pool
    recall = len(gold_in_pool) / n_gold
    all_gold = len(gold_in_pool) == len(gold_titles) and len(gold_titles) > 0

    rank, mrr = compute_first_gold_rank(pool_titles, gold_titles)

    trace = record.get("trace", [])
    if trace and paradigm == "E-RAG-Balanced":
        # E-RAG explicit hop progression from recorded belief-graph trace
        h0_pool = trace[0].get("pool", [])
        h0_gold = set(h0_pool) & gold_titles
        hop0_recall = len(h0_gold) / n_gold
        hop0_precision = len(h0_gold) / max(len(h0_pool), 1)

        if len(trace) > 1:
            h1_pool = trace[1].get("pool", [])
            h1_gold = set(h1_pool) & gold_titles
            hop1_recall = len(h1_gold) / n_gold
            hop1_precision = len(h1_gold) / max(len(h1_pool), 1)
        else:
            hop1_recall = hop0_recall
            hop1_precision = hop0_precision

        if len(trace) > 2:
            h2_pool = trace[2].get("pool", [])
            h2_gold = set(h2_pool) & gold_titles
            hop2_recall = len(h2_gold) / n_gold
            hop2_precision = len(h2_gold) / max(len(h2_pool), 1)
        else:
            hop2_recall = hop1_recall
            hop2_precision = hop1_precision

        multi_hop_gain = round(recall - hop0_recall, 4)
        bridge_resolved = bool(hop0_recall < 1.0 and recall == 1.0)
    elif paradigm == "IRCoT-Controlled":
        # Initial retrieval in IRCoT is top-k=3
        h0_pool = pool_titles[:3]
        h0_gold = set(h0_pool) & gold_titles
        hop0_recall = len(h0_gold) / n_gold
        hop0_precision = len(h0_gold) / max(len(h0_pool), 1)
        hop1_recall = recall
        hop1_precision = precision
        hop2_recall = recall
        hop2_precision = precision
        multi_hop_gain = round(recall - hop0_recall, 4)
        bridge_resolved = bool(hop0_recall < 1.0 and recall == 1.0)
    else:
        # Standard RAG (single hop)
        hop0_recall = recall
        hop0_precision = precision
        hop1_recall = recall
        hop1_precision = precision
        hop2_recall = recall
        hop2_precision = precision
        multi_hop_gain = 0.0
        bridge_resolved = False

    enriched = dict(record)
    enriched.update({
        "retrieval_precision": round(precision, 4),
        "retrieval_recall": round(recall, 4),
        "all_gold": all_gold,
        "first_gold_rank": rank,
        "mrr": round(mrr, 4),
        "hop0_recall": round(hop0_recall, 4),
        "hop0_precision": round(hop0_precision, 4),
        "hop1_recall": round(hop1_recall, 4),
        "hop1_precision": round(hop1_precision, 4),
        "hop2_recall": round(hop2_recall, 4),
        "hop2_precision": round(hop2_precision, 4),
        "multi_hop_recall_gain": multi_hop_gain,
        "bridge_resolved": bridge_resolved,
    })
    return enriched


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)

    print(f"Loading gold manifest: {MANIFEST_PATH}")
    manifest = load_manifest(MANIFEST_PATH)
    print(f"Loaded {len(manifest)} questions from manifest.")

    summary_rows = []

    for paradigm, retriever_label, src_path, dst_filename in CONDITIONS:
        if not os.path.exists(src_path):
            print(f"WARNING: Source file missing: {src_path}")
            continue

        records = []
        with open(src_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        enriched_records = []
        for r in records:
            q_id = r.get("id", r.get("_id"))
            golds = manifest.get(q_id, set())
            enriched_records.append(enrich_record(r, golds, paradigm))

        dst_path = os.path.join(OUT_DIR, dst_filename)
        with open(dst_path, "w", encoding="utf-8") as f:
            for er in enriched_records:
                f.write(json.dumps(er, ensure_ascii=False) + "\n")

        print(f"Enriched {len(enriched_records)} records -> {dst_path}")

        # Compute summary aggregations
        n = len(enriched_records)
        em = np.mean([r.get("em", 0.0) for r in enriched_records])
        f1 = np.mean([r.get("f1", 0.0) for r in enriched_records])
        prec = np.mean([r["retrieval_precision"] for r in enriched_records])
        rec = np.mean([r["retrieval_recall"] for r in enriched_records])
        all_gold = np.mean([1.0 if r["all_gold"] else 0.0 for r in enriched_records])
        docs = np.mean([len(r.get("pool_titles", [])) for r in enriched_records])
        mrr = np.mean([r["mrr"] for r in enriched_records])
        
        h0_rec = np.mean([r["hop0_recall"] for r in enriched_records])
        h1_rec = np.mean([r["hop1_recall"] for r in enriched_records])
        h2_rec = np.mean([r["hop2_recall"] for r in enriched_records])
        hop_gain = np.mean([r["multi_hop_recall_gain"] for r in enriched_records])
        bridge_res = np.mean([1.0 if r["bridge_resolved"] else 0.0 for r in enriched_records])

        # Cost
        total_cost = 0.0
        calls = [r.get("llm_calls", 0) for r in enriched_records]
        for r in enriched_records:
            total_cost += (r.get("in_tokens", 0) * 0.059 + r.get("out_tokens", 0) * 0.079) / 1_000_000.0
        cost_per_1k = (total_cost / n) * 1000.0 if n > 0 else 0.0

        summary_rows.append({
            "Paradigm": paradigm,
            "Retriever": retriever_label,
            "N": n,
            "Exact Match": round(float(em), 4),
            "Token F1": round(float(f1), 4),
            "Final Recall": round(float(rec), 4),
            "All-Gold Rate": round(float(all_gold), 4),
            "Retrieval Precision": round(float(prec), 4),
            "Mean Docs": round(float(docs), 2),
            "MRR": round(float(mrr), 4),
            "Hop 0 Recall": round(float(h0_rec), 4),
            "Hop 1 Recall": round(float(h1_rec), 4),
            "Hop 2 Recall": round(float(h2_rec), 4),
            "Multi-Hop Recall Gain": round(float(hop_gain), 4),
            "Bridge Resolved Rate": round(float(bridge_res), 4),
            "Mean Calls": round(float(np.mean(calls)), 2),
            "Cost / 1k USD": round(float(cost_per_1k), 4),
        })

    df = pd.DataFrame(summary_rows)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSuccessfully generated summary CSV: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
