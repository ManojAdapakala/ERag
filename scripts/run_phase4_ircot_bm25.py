"""Phase 4 Benchmark Harness: BM25 Lexical vs MiniLM Dense Retrieval across Systems.

Evaluates on HotpotQA Test-400 (data/manifests/test.jsonl):
1. Standard RAG + BM25 (results/bm25/standard_bm25_test.jsonl)
2. IRCoT-Controlled + BM25 (results/bm25/ircot_bm25_test.jsonl)
3. E-RAG-Balanced + BM25 (results/bm25/erag_balanced_bm25_test.jsonl)

Integrates matched Dense (all-MiniLM-L6-v2) baselines:
4. Standard RAG + MiniLM (results/standard_test.jsonl)
5. IRCoT-Controlled + MiniLM (results/ircot_test.jsonl)
6. E-RAG-Balanced + MiniLM (results/cost_opt/erag_balanced_test.jsonl)

Outputs:
- research/ircot/bm25_vs_dense_benchmark.csv
"""

import atexit
import copy
import ctypes
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_cfg
from src.eval.metrics import exact_match, f1_score, is_unknown, retrieval_recall, auroc, ece_by_level, brier, normalize_answer
from src.llm.client import LLM, Tracker
from src.retrieval.index import RetrieverFactory
from src.epistemic.nli import NLI
from src.systems.standard import run_standard
from src.systems.ircot import run_ircot
from src.systems.erag import run_erag


def is_pid_running(pid: int) -> bool:
    """Check if process ID is alive on Windows."""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if h:
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    return False


def acquire_process_lock(lock_path: str = "results/bm25/.runner.lock"):
    """Enforce single-instance execution to prevent duplicate processes."""
    if os.path.exists(lock_path):
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                old_pid = int(f.read().strip())
            if is_pid_running(old_pid):
                print(f"[PROCESS LOCK] Another runner with PID {old_pid} is active! Exiting.")
                sys.exit(0)
        except Exception:
            pass

    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    def release_lock():
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except Exception:
            pass

    atexit.register(release_lock)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_records(records: List[Dict[str, Any]], system_name: str, retriever_type: str) -> Dict[str, Any]:
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
    hops_list = [r.get("hops", 0) for r in records]
    wall_lat = [r.get("wall_latency", 0.0) for r in records]
    
    pool_docs = [len(r.get("pool_titles", [])) for r in records]
    cost_usd_list = [(i * 0.10 + o * 0.20) / 1e6 for i, o in zip(in_tok, out_tok)]
    
    return {
        "system": system_name,
        "retriever": retriever_type,
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
        "mean_total_tokens": float(np.mean(tot_tok)),
        "cost_per_1k_usd": float(np.mean(cost_usd_list)) * 1000.0,
        "mean_wall_latency": float(np.mean(wall_lat)),
    }


def execute_runner(
    system_type: str,
    items: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    nli: Optional[NLI],
    out_path: str,
    desc: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute a system with BM25 retriever and save predictions."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    factory = RetrieverFactory(setting="distractor", retriever_type="bm25")
    llm = LLM(
        cfg,
        use_cache=cfg["run"]["cache"],
        namespace=cfg["run"]["cache_namespace"],
    )

    records: List[Dict[str, Any]] = []
    seen_ids = set()

    if os.path.exists(out_path):
        raw_records = load_jsonl(out_path)
        records = []
        for r in raw_records:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                records.append(r)
        if len(records) >= len(items):
            print(f"Loaded existing results from {out_path} ({len(records)} records)")
            return records, evaluate_records(records, system_type, "BM25")
        else:
            print(f"Resuming evaluation: {len(records)}/{len(items)} items processed in {out_path}")

    remaining = [it for it in items if it["_id"] not in seen_ids]
    with open(out_path, "a", encoding="utf-8") as f:
        for item in tqdm(remaining, desc=desc, initial=len(records), total=len(items), file=sys.stdout):
            tr = Tracker()
            t0 = time.time()
            retriever = factory.for_item(item)

            try:
                if system_type == "Standard RAG":
                    r = run_standard(item, retriever, llm, cfg, tr)
                elif system_type == "IRCoT":
                    r = run_ircot(item, retriever, llm, cfg, tr)
                elif system_type == "E-RAG-Balanced":
                    r = run_erag(item, retriever, llm, nli, cfg, tr)
                else:
                    raise ValueError(f"Unknown system type: {system_type}")
            except Exception as err:
                print(f"\nWarning: Exception on question {item['_id']}: {err}. Retrying in 5s...")
                time.sleep(5.0)
                if system_type == "Standard RAG":
                    r = run_standard(item, retriever, llm, cfg, tr)
                elif system_type == "IRCoT":
                    r = run_ircot(item, retriever, llm, cfg, tr)
                elif system_type == "E-RAG-Balanced":
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

    summary = evaluate_records(records, system_type, "BM25")
    return records, summary


def main():
    acquire_process_lock()
    print("=================================================================")
    print("PHASE 4: IRCOT METHODOLOGY AUDIT & FAIR BM25 BENCHMARK (TEST-400)")
    print("=================================================================")

    test_path = "data/manifests/test.jsonl"
    test_items = load_jsonl(test_path)
    print(f"Loaded Test Manifest: {len(test_items)} questions.")

    os.makedirs("results/bm25", exist_ok=True)
    os.makedirs("research/ircot", exist_ok=True)

    # 1. Base configs
    base_cfg = load_cfg("configs/base.yaml", "configs/final.yaml")
    base_cfg["data"]["split"] = "test"
    base_cfg["run"]["cache"] = True
    base_cfg["run"]["cache_namespace"] = "test_v1"

    print("Initializing NLI Verifier (cross-encoder/nli-deberta-v3-base)...")
    nli = NLI(base_cfg["nli"]["model"], base_cfg["nli"]["window"])

    # 2. Benchmark 1: Standard RAG + BM25
    print("\n--- Running 1/3: Standard RAG + BM25 ---")
    std_cfg = copy.deepcopy(base_cfg)
    std_out = "results/bm25/standard_bm25_test.jsonl"
    _, std_bm25_metrics = execute_runner(
        "Standard RAG", test_items, std_cfg, None, std_out, "Standard RAG + BM25"
    )

    # 3. Benchmark 2: IRCoT-Controlled + BM25
    print("\n--- Running 2/3: IRCoT-Controlled + BM25 ---")
    ircot_cfg = copy.deepcopy(base_cfg)
    ircot_out = "results/bm25/ircot_bm25_test.jsonl"
    _, ircot_bm25_metrics = execute_runner(
        "IRCoT", test_items, ircot_cfg, None, ircot_out, "IRCoT-Controlled + BM25"
    )

    # 4. Benchmark 3: E-RAG-Balanced + BM25
    print("\n--- Running 3/3: E-RAG-Balanced + BM25 ---")
    erag_cfg = load_cfg("configs/base.yaml", "configs/erag_balanced.yaml")
    erag_cfg["data"]["split"] = "test"
    erag_cfg["run"]["cache"] = True
    erag_cfg["run"]["cache_namespace"] = "test_v1"
    erag_out = "results/bm25/erag_balanced_bm25_test.jsonl"
    _, erag_bm25_metrics = execute_runner(
        "E-RAG-Balanced", test_items, erag_cfg, nli, erag_out, "E-RAG-Balanced + BM25"
    )

    # 5. Load Matched Dense MiniLM Baselines
    print("\n--- Ingesting Dense MiniLM Baselines for Direct Comparison ---")
    std_dense_recs = load_jsonl("results/standard_test.jsonl")
    ircot_dense_recs = load_jsonl("results/ircot_test.jsonl")
    erag_dense_recs = load_jsonl("results/cost_opt/erag_balanced_test.jsonl")

    std_dense_metrics = evaluate_records(std_dense_recs, "Standard RAG", "MiniLM (Dense)")
    ircot_dense_metrics = evaluate_records(ircot_dense_recs, "IRCoT", "MiniLM (Dense)")
    erag_dense_metrics = evaluate_records(erag_dense_recs, "E-RAG-Balanced", "MiniLM (Dense)")

    # 6. Assemble Comparative Matrix
    all_metrics = [
        std_bm25_metrics,
        std_dense_metrics,
        ircot_bm25_metrics,
        ircot_dense_metrics,
        erag_bm25_metrics,
        erag_dense_metrics,
    ]
    df = pd.DataFrame(all_metrics)

    cols_order = [
        "system", "retriever", "n", "em", "f1", "gold_recall", "all_gold_rate",
        "mean_pool_docs", "coverage", "selective_em", "ece", "brier", "auroc",
        "mean_llm_calls", "mean_total_tokens", "cost_per_1k_usd", "mean_wall_latency"
    ]
    df = df[[c for c in cols_order if c in df.columns]]

    csv_path = "research/ircot/bm25_vs_dense_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved BM25 vs Dense Benchmark Matrix to: {csv_path}")

    print("\n================== BM25 VS DENSE COMPARATIVE MATRIX ==================")
    disp_cols = ["system", "retriever", "em", "f1", "gold_recall", "all_gold_rate", "selective_em", "mean_llm_calls", "cost_per_1k_usd"]
    print(df[disp_cols].to_string(index=False))

    print("\nPHASE 4 BENCHMARK EXECUTION COMPLETE.")


if __name__ == "__main__":
    main()

