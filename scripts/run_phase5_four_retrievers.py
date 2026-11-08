"""Phase 5 Benchmark Harness: Four-Retriever Comparative Benchmark across Multi-Hop Systems.

Evaluates on HotpotQA Test-400 (data/manifests/test.jsonl) across 4 retrieval models:
1. BM25 (BM25Plus lexical)
2. MiniLM (sentence-transformers/all-MiniLM-L6-v2 dense 384d)
3. E5 (intfloat/e5-base-v2 dense 768d asymmetric)
4. BGE (BAAI/bge-base-en-v1.5 dense 768d query-instructed)

Systems evaluated per retriever:
- Standard RAG
- IRCoT-Controlled
- E-RAG-Balanced

Total: 12 experimental cells. Re-uses existing completed runs (BM25 from Phase 4, MiniLM from Phase 3).
Outputs:
- research/retrievers/four_retriever_benchmark.csv
"""

import argparse
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
from src.retrieval.index import RetrieverFactory, Embedder, MODEL_ALIASES
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


def acquire_process_lock(lock_path: str = "results/phase5/.runner.lock"):
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


def evaluate_records(records: List[Dict[str, Any]], system_name: str, retriever_name: str) -> Dict[str, Any]:
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
        auroc_val = auroc(confs, em_binary)
    except Exception:
        auroc_val = float("nan")

    # Document accumulation count
    mean_docs = float(np.mean([len(r.get("pool_titles", [])) for r in records])) if records else 0.0

    calls = [r.get("llm_calls", 0) for r in records]
    tokens = [r.get("in_tokens", 0) + r.get("out_tokens", 0) for r in records]
    
    # Cost calculation: llama-3.3-70b-versatile ($0.059 per 1M input, $0.079 per 1M output)
    total_cost = 0.0
    for r in records:
        c = (r.get("in_tokens", 0) * 0.059 + r.get("out_tokens", 0) * 0.079) / 1_000_000.0
        total_cost += c
    cost_per_1k = (total_cost / len(records)) * 1000.0 if records else 0.0

    return {
        "System": system_name,
        "Retriever": retriever_name,
        "N": len(records),
        "Exact Match": round(float(np.mean(em_list)), 4),
        "Token F1": round(float(np.mean(f1_list)), 4),
        "Gold Para Recall": round(float(np.mean(rec_list)), 4),
        "All-Gold Rate": round(float(np.mean(all_gold_list)), 4),
        "Mean Docs": round(mean_docs, 2),
        "Coverage": round(float(np.mean(cov_list)), 4),
        "Selective EM": round(selective_em, 4),
        "ECE": round(float(ece_val), 4),
        "Brier": round(float(brier_val), 4),
        "AUROC": round(float(auroc_val), 4) if not np.isnan(auroc_val) else "N/A",
        "Mean Calls": round(float(np.mean(calls)), 2),
        "Mean Tokens": round(float(np.mean(tokens)), 1),
        "Cost / 1k USD": round(cost_per_1k, 4),
    }


def execute_system_retriever(
    system_type: str,
    retriever_name: str,
    output_path: str,
    test_items: List[Dict[str, Any]],
    nli: Optional[NLI],
    cfg: Dict[str, Any],
    embedder: Optional[Embedder] = None,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    existing_records = load_jsonl(output_path)
    done_ids = {r["id"] for r in existing_records}

    target_items = test_items[:limit] if limit else test_items
    remaining = [it for it in target_items if it["_id"] not in done_ids]

    print(f"\n[{system_type} + {retriever_name}] Target: {len(target_items)} | Completed: {len(done_ids)} | Remaining: {len(remaining)}")

    records = list(existing_records)
    if remaining:
        # Create RetrieverFactory for this retriever
        factory = RetrieverFactory(
            setting="distractor",
            embedder=embedder,
            retriever_type=retriever_name,
        )
        llm = LLM(
            cfg,
            use_cache=cfg["run"].get("cache", True),
            namespace=cfg["run"].get("cache_namespace", "test_v1"),
        )

        with open(output_path, "a", encoding="utf-8") as f:
            for item in tqdm(remaining, desc=f"Evaluating {system_type} + {retriever_name}", initial=len(records), total=len(target_items), file=sys.stdout):
                retriever = factory.for_item(item)
                tr = Tracker()
                t0 = time.time()

                try:
                    if system_type == "Standard RAG":
                        r = run_standard(item, retriever, llm, cfg, tr)
                    elif system_type == "IRCoT":
                        r = run_ircot(item, retriever, llm, cfg, tr)
                    elif system_type == "E-RAG-Balanced":
                        r = run_erag(item, retriever, llm, nli, cfg, tr)
                    else:
                        raise ValueError(f"Unknown system: {system_type}")
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

    summary = evaluate_records(records, system_type, retriever_name)
    return records, summary


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Four-Retriever Comparative Benchmark")
    parser.add_argument("--retriever", type=str, default="all", choices=["all", "e5", "bge", "aggregate-only"])
    parser.add_argument("--system", type=str, default="all", choices=["all", "standard", "ircot", "erag"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation questions")
    args = parser.parse_args()

    acquire_process_lock()
    print("=================================================================")
    print("PHASE 5: FOUR-RETRIEVER COMPARATIVE BENCHMARK (TEST-400)")
    print("=================================================================")

    test_path = "data/manifests/test.jsonl"
    test_items = load_jsonl(test_path)
    print(f"Loaded Test Manifest: {len(test_items)} questions.")

    os.makedirs("results/phase5", exist_ok=True)
    os.makedirs("research/retrievers", exist_ok=True)

    # 1. Base configs
    base_cfg = load_cfg("configs/base.yaml", "configs/final.yaml")
    base_cfg["data"]["split"] = "test"
    base_cfg["run"]["cache"] = True
    base_cfg["run"]["cache_namespace"] = "test_v1"

    # Define balanced config for E-RAG
    erag_balanced_cfg = load_cfg("configs/base.yaml", "configs/erag_balanced.yaml")
    erag_balanced_cfg["data"]["split"] = "test"
    erag_balanced_cfg["run"]["cache"] = True
    erag_balanced_cfg["run"]["cache_namespace"] = "test_v1"

    all_summaries: List[Dict[str, Any]] = []

    # 2. Existing MiniLM Runs
    minilm_files = [
        ("Standard RAG", "MiniLM (Dense)", "results/standard_test.jsonl"),
        ("IRCoT", "MiniLM (Dense)", "results/ircot_test.jsonl"),
        ("E-RAG-Balanced", "MiniLM (Dense)", "results/cost_opt/erag_balanced_test.jsonl"),
    ]
    for sys_name, ret_name, fpath in minilm_files:
        if os.path.exists(fpath):
            recs = load_jsonl(fpath)
            if recs:
                s = evaluate_records(recs, sys_name, ret_name)
                all_summaries.append(s)

    # 3. Existing BM25 Runs
    bm25_files = [
        ("Standard RAG", "BM25 (Sparse)", "results/bm25/standard_bm25_test.jsonl"),
        ("IRCoT", "BM25 (Sparse)", "results/bm25/ircot_bm25_test.jsonl"),
        ("E-RAG-Balanced", "BM25 (Sparse)", "results/bm25/erag_balanced_bm25_test.jsonl"),
    ]
    for sys_name, ret_name, fpath in bm25_files:
        if os.path.exists(fpath):
            recs = load_jsonl(fpath)
            if recs:
                s = evaluate_records(recs, sys_name, ret_name)
                all_summaries.append(s)

    # 4. Existing E5 Runs
    e5_files = [
        ("Standard RAG", "E5-base-v2 (Dense)", "results/phase5/standard_e5_test.jsonl"),
        ("IRCoT", "E5-base-v2 (Dense)", "results/phase5/ircot_e5_test.jsonl"),
        ("E-RAG-Balanced", "E5-base-v2 (Dense)", "results/phase5/erag_balanced_e5_test.jsonl"),
    ]
    for sys_name, ret_name, fpath in e5_files:
        if os.path.exists(fpath):
            recs = load_jsonl(fpath)
            if recs:
                s = evaluate_records(recs, sys_name, ret_name)
                all_summaries.append(s)

    # 5. Existing BGE Runs
    bge_files = [
        ("Standard RAG", "BGE-base-en-v1.5 (Dense)", "results/phase5/standard_bge_test.jsonl"),
        ("IRCoT", "BGE-base-en-v1.5 (Dense)", "results/phase5/ircot_bge_test.jsonl"),
        ("E-RAG-Balanced", "BGE-base-en-v1.5 (Dense)", "results/phase5/erag_balanced_bge_test.jsonl"),
    ]
    for sys_name, ret_name, fpath in bge_files:
        if os.path.exists(fpath):
            recs = load_jsonl(fpath)
            if recs:
                s = evaluate_records(recs, sys_name, ret_name)
                all_summaries.append(s)

    if args.retriever != "aggregate-only":
        print("Initializing NLI Verifier (cross-encoder/nli-deberta-v3-base)...")
        nli = NLI(base_cfg["nli"]["model"], base_cfg["nli"]["window"])

        retrievers_to_run = []
        if args.retriever in ["all", "e5"]:
            retrievers_to_run.append(("e5", "intfloat/e5-base-v2", "E5-base-v2 (Dense)"))
        if args.retriever in ["all", "bge"]:
            retrievers_to_run.append(("bge", "BAAI/bge-base-en-v1.5", "BGE-base-en-v1.5 (Dense)"))

        for ret_alias, model_id, ret_label in retrievers_to_run:
            print(f"\n==================================================")
            print(f"LOADING EMBEDDER: {model_id} ({ret_label})")
            print(f"==================================================")
            embedder = Embedder(model_id)

            systems = []
            if args.system in ["all", "standard"]:
                systems.append(("Standard RAG", f"results/phase5/standard_{ret_alias}_test.jsonl", base_cfg))
            if args.system in ["all", "ircot"]:
                systems.append(("IRCoT", f"results/phase5/ircot_{ret_alias}_test.jsonl", base_cfg))
            if args.system in ["all", "erag"]:
                systems.append(("E-RAG-Balanced", f"results/phase5/erag_balanced_{ret_alias}_test.jsonl", erag_balanced_cfg))

            for sys_name, out_path, run_cfg in systems:
                recs, summary = execute_system_retriever(
                    system_type=sys_name,
                    retriever_name=ret_label,
                    output_path=out_path,
                    test_items=test_items,
                    nli=nli,
                    cfg=run_cfg,
                    embedder=embedder,
                    limit=args.limit,
                )
                all_summaries.append(summary)

    # Compile Full 12-cell Comparative DataFrame
    df = pd.DataFrame(all_summaries)
    # Deduplicate in case existing and newly run overlap
    df = df.drop_duplicates(subset=["System", "Retriever"], keep="last")
    
    # Sort logically: Systems Standard -> IRCoT -> E-RAG, Retrievers BM25 -> MiniLM -> E5 -> BGE
    sys_order = {"Standard RAG": 0, "IRCoT": 1, "E-RAG-Balanced": 2}
    df["sys_rank"] = df["System"].map(lambda x: sys_order.get(x, 99))
    df = df.sort_values(by=["sys_rank", "Retriever"]).drop(columns=["sys_rank"])

    out_csv = "research/retrievers/four_retriever_benchmark.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved 12-System Benchmark Matrix to: {out_csv}")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
