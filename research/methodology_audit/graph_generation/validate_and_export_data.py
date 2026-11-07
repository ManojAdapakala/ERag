"""Validate raw experiment outputs across all five systems on Test-400 and export validated plot data."""

import hashlib
import json
import os
import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.eval.metrics import brier, exact_match, f1_score, is_unknown, normalize_answer, retrieval_recall
from src.eval.stats import mcnemar_exact, paired_bootstrap


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_continuous_ece(conf: List[float], correct: List[int], n_bins: int = 10) -> float:
    """Compute standard continuous 10-bin Expected Calibration Error."""
    if not conf or not correct:
        return 0.0
    c_arr = np.asarray(conf, dtype=float)
    y_arr = np.asarray(correct, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(c_arr)
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (c_arr >= low) & (c_arr <= high if i == n_bins - 1 else c_arr < high)
        if np.any(mask):
            bin_acc = np.mean(y_arr[mask])
            bin_conf = np.mean(c_arr[mask])
            bin_weight = np.sum(mask) / n
            ece += bin_weight * abs(bin_acc - bin_conf)
    return float(ece)


def main():
    manifest_path = "data/manifests/test.jsonl"
    results_dir = "results"
    out_dir = "research/methodology_audit/validated_plot_data"
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== [Data Validation] Loading Manifest: {manifest_path} ===")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_items = [json.loads(line) for line in f if line.strip()]
    
    n_manifest = len(manifest_items)
    assert n_manifest == 400, f"Expected 400 manifest items, found {n_manifest}"
    manifest_ids = [item["_id"] for item in manifest_items]
    manifest_dict = {item["_id"]: item for item in manifest_items}
    print(f"Loaded {n_manifest} manifest items. Manifest SHA-256: {sha256_file(manifest_path)}")

    systems = {
        "Closed-Book": "closed_book_test.jsonl",
        "Standard RAG": "standard_test.jsonl",
        "IRCoT": "ircot_test.jsonl",
        "E-RAG Observe": "erag_observe_test.jsonl",
        "E-RAG Active": "erag_active_test.jsonl",
    }

    raw_data = {}
    file_hashes = {}

    for sys_name, fname in systems.items():
        fpath = os.path.join(results_dir, fname)
        assert os.path.exists(fpath), f"Source file does not exist: {fpath}"
        fhash = sha256_file(fpath)
        file_hashes[sys_name] = (fpath, fhash)

        with open(fpath, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        print(f"Loaded {sys_name}: {len(records)} records from {fname} (SHA-256: {fhash[:12]}...)")
        assert len(records) == 400, f"Expected 400 records for {sys_name}, got {len(records)}"

        record_ids = [r["id"] for r in records]
        assert record_ids == manifest_ids, f"ID sequence mismatch in {sys_name} vs manifest"
        raw_data[sys_name] = records

    print("All 5 systems validated: 400 records each, zero ID mismatches, 100% order fidelity.\n")

    # Construct unified per-question DataFrame
    per_q_rows = []
    for idx, qid in enumerate(manifest_ids):
        m_item = manifest_dict[qid]
        q_text = m_item["question"]
        gold_ans = m_item["answer"]
        q_type = m_item.get("type", "unknown")
        q_level = m_item.get("level", "hard")

        row = {
            "index": idx,
            "id": qid,
            "question": q_text,
            "gold_answer": gold_ans,
            "type": q_type,
            "level": q_level,
        }

        for sys_name, records in raw_data.items():
            r = records[idx]
            prefix = sys_name.lower().replace(" ", "_").replace("-", "_")
            ans = r.get("answer", "")
            
            # Recalculate metrics for independent verification
            em = exact_match(ans, gold_ans)
            f1 = f1_score(ans, gold_ans)
            unknown = is_unknown(ans)
            
            # Recall
            if sys_name == "Closed-Book":
                rec, all_g = np.nan, np.nan
            else:
                pool_titles = r.get("pool_titles", [])
                rec, all_g = retrieval_recall(pool_titles, m_item)

            row[f"{prefix}__answer"] = ans
            row[f"{prefix}__em"] = em
            row[f"{prefix}__f1"] = f1
            row[f"{prefix}__unknown"] = unknown
            row[f"{prefix}__recall"] = rec
            row[f"{prefix}__all_gold"] = all_g
            row[f"{prefix}__llm_calls"] = r.get("llm_calls", 1)
            row[f"{prefix}__retrievals"] = r.get("retrievals", 0 if sys_name == "Closed-Book" else 1)
            row[f"{prefix}__in_tokens"] = r.get("in_tokens", 0)
            row[f"{prefix}__out_tokens"] = r.get("out_tokens", 0)
            row[f"{prefix}__total_tokens"] = r.get("in_tokens", 0) + r.get("out_tokens", 0)
            row[f"{prefix}__api_latency"] = r.get("api_latency", 0.0)
            row[f"{prefix}__wall_latency"] = r.get("wall_latency", 0.0)

            # Epistemic fields
            if "answer_conf" in r:
                row[f"{prefix}__answer_conf"] = r["answer_conf"]
            elif "c0" in r:
                row[f"{prefix}__answer_conf"] = r["c0"]
            else:
                row[f"{prefix}__answer_conf"] = np.nan

            if "c0" in r:
                row[f"{prefix}__c0"] = r["c0"]
            else:
                row[f"{prefix}__c0"] = np.nan

        per_q_rows.append(row)

    df_per_q = pd.DataFrame(per_q_rows)
    per_q_csv = os.path.join(out_dir, "test400_five_systems_per_question.csv")
    df_per_q.to_csv(per_q_csv, index=False)
    print(f"Saved validated per-question dataset: {per_q_csv} ({len(df_per_q)} rows)")

    # Compute aggregate summary table
    agg_rows = []
    for sys_name in systems:
        prefix = sys_name.lower().replace(" ", "_").replace("-", "_")
        ems = df_per_q[f"{prefix}__em"].values
        f1s = df_per_q[f"{prefix}__f1"].values
        unknowns = df_per_q[f"{prefix}__unknown"].values
        recalls = df_per_q[f"{prefix}__recall"].dropna().values
        all_golds = df_per_q[f"{prefix}__all_gold"].dropna().values
        
        answered_mask = ~unknowns
        n_ans = np.sum(answered_mask)
        coverage = n_ans / 400.0
        sel_em = np.mean(ems[answered_mask]) if n_ans > 0 else 0.0

        confs = df_per_q[f"{prefix}__answer_conf"].values
        has_conf = not np.all(np.isnan(confs))
        if has_conf:
            ece_10 = compute_continuous_ece(confs.tolist(), ems.astype(int).tolist(), n_bins=10)
            brier_score = float(np.mean((confs - ems) ** 2))
        else:
            ece_10 = np.nan
            brier_score = np.nan

        wall_lats = df_per_q[f"{prefix}__wall_latency"].values
        total_tokens = df_per_q[f"{prefix}__total_tokens"].values
        llm_calls = df_per_q[f"{prefix}__llm_calls"].values
        rets = df_per_q[f"{prefix}__retrievals"].values

        agg_rows.append({
            "System": sys_name,
            "N": 400,
            "Exact Match": float(np.mean(ems)),
            "Token F1": float(np.mean(f1s)),
            "Gold Paragraph Recall": float(np.mean(recalls)) if len(recalls) > 0 else np.nan,
            "All-Gold Rate": float(np.mean(all_golds)) if len(all_golds) > 0 else np.nan,
            "Abstention Rate": float(np.mean(unknowns)),
            "Coverage": float(coverage),
            "Selective EM": float(sel_em),
            "ECE (10-bin)": ece_10,
            "Brier Score": brier_score,
            "Mean LLM Calls/Q": float(np.mean(llm_calls)),
            "Mean Retrievals/Q": float(np.mean(rets)),
            "Mean Wall Latency (s)": float(np.mean(wall_lats)),
            "Median Wall Latency (s)": float(np.median(wall_lats)),
            "P90 Wall Latency (s)": float(np.quantile(wall_lats, 0.90)),
            "P95 Wall Latency (s)": float(np.quantile(wall_lats, 0.95)),
            "Mean Total Tokens": float(np.mean(total_tokens)),
        })

    df_agg = pd.DataFrame(agg_rows)
    agg_csv = os.path.join(out_dir, "test400_aggregate_metrics.csv")
    df_agg.to_csv(agg_csv, index=False)
    print(f"Saved aggregate metrics table: {agg_csv}")
    print(df_agg.to_string(index=False))

    # Dedicated IRCoT vs E-RAG Active Paired Table
    paired_rows = []
    ircot_prefix = "ircot"
    active_prefix = "e_rag_active"
    for idx, qid in enumerate(manifest_ids):
        ircot_em = df_per_q.loc[idx, f"{ircot_prefix}__em"]
        active_em = df_per_q.loc[idx, f"{active_prefix}__em"]
        ircot_f1 = df_per_q.loc[idx, f"{ircot_prefix}__f1"]
        active_f1 = df_per_q.loc[idx, f"{active_prefix}__f1"]
        ircot_rec = df_per_q.loc[idx, f"{ircot_prefix}__recall"]
        active_rec = df_per_q.loc[idx, f"{active_prefix}__recall"]
        ircot_lat = df_per_q.loc[idx, f"{ircot_prefix}__wall_latency"]
        active_lat = df_per_q.loc[idx, f"{active_prefix}__wall_latency"]
        ircot_tok = df_per_q.loc[idx, f"{ircot_prefix}__total_tokens"]
        active_tok = df_per_q.loc[idx, f"{active_prefix}__total_tokens"]

        paired_rows.append({
            "id": qid,
            "ircot_em": ircot_em,
            "active_em": active_em,
            "delta_em": active_em - ircot_em,
            "ircot_f1": ircot_f1,
            "active_f1": active_f1,
            "delta_f1": active_f1 - ircot_f1,
            "ircot_recall": ircot_rec,
            "active_recall": active_rec,
            "delta_recall": active_rec - ircot_rec,
            "ircot_wall_latency": ircot_lat,
            "active_wall_latency": active_lat,
            "delta_wall_latency": active_lat - ircot_lat,
            "ircot_total_tokens": ircot_tok,
            "active_total_tokens": active_tok,
            "delta_total_tokens": active_tok - ircot_tok,
        })

    df_paired = pd.DataFrame(paired_rows)
    paired_csv = os.path.join(out_dir, "ircot_vs_active_paired.csv")
    df_paired.to_csv(paired_csv, index=False)
    print(f"Saved paired comparison dataset: {paired_csv} ({len(df_paired)} rows)\n")


if __name__ == "__main__":
    main()

