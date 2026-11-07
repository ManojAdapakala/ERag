"""Isolated Efficiency & Latency Profiling Harness for E-RAG Benchmarking (Stage 3).

Measures granular sub-component latencies on HotpotQA Dev-100 using monotonic high-resolution clocks.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_cfg
from src.llm.client import LLM, Tracker
from src.retrieval.index import Embedder, RetrieverFactory
from src.epistemic.nli import NLI
from src.eval.metrics import exact_match, f1_score, is_unknown, retrieval_recall
from src.systems.standard import run_standard, run_closed_book
from src.systems.ircot import run_ircot
from src.systems.erag import run_erag, propose_queries, build_graph, conf_value, extract_claims, self_consistency, targeted_retrieve
from src.llm.prompts import answer_messages


def profile_question_erag(
    item: Dict[str, Any],
    factory: RetrieverFactory,
    llm: LLM,
    nli: NLI,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute E-RAG with microsecond-level sub-component instrumentation."""
    e = cfg.get("erag", {})
    k = cfg["retrieval"].get("k", 3)
    q = item["question"]
    ablate = set(e.get("ablate", []))
    use_status = "no_nli" not in ablate

    tracker = Tracker()
    timings = {}
    
    t_start = time.perf_counter()

    # 1. Initial retrieval
    t0 = time.perf_counter()
    retriever = factory.for_item(item)
    pool = retriever.search(q, k=k)
    timings["t_init_retrieval"] = time.perf_counter() - t0
    tracker.retrievals += 1

    timings["t_subseq_retrieval"] = 0.0
    timings["t_llm_draft"] = 0.0
    timings["t_sc"] = 0.0
    timings["t_claims"] = 0.0
    timings["t_nli"] = 0.0
    timings["t_graph"] = 0.0
    timings["t_query_gen"] = 0.0
    timings["t_memory_read"] = 0.0
    timings["t_memory_write"] = 0.0

    nli_pairs_count = 0
    total_claims_count = 0
    total_edges_count = 0

    hop, c0, trace = 0, None, []

    while True:
        # 2. Draft generation
        t0 = time.perf_counter()
        draft = llm.chat(answer_messages(q, pool), tracker, temperature=0.0)
        timings["t_llm_draft"] += time.perf_counter() - t0

        # 3. Self-consistency
        if c0 is None:
            t0 = time.perf_counter()
            sc = self_consistency(llm, q, pool, draft, tracker, k=e.get("sc_samples", 5), temperature=e.get("sc_temperature", 0.7))
            c0 = conf_value(sc, cfg)
            timings["t_sc"] += time.perf_counter() - t0

        # 4. Claim extraction
        t0 = time.perf_counter()
        claims = extract_claims(llm, q, draft, pool, tracker, max_claims=e.get("max_claims", 5))
        timings["t_claims"] += time.perf_counter() - t0
        total_claims_count += len(claims)

        # 5. NLI Verification
        t0 = time.perf_counter()
        for c in claims:
            for p in pool:
                nli.best(c["text"], p)
                nli_pairs_count += 1
        t_nli_raw = time.perf_counter() - t0
        timings["t_nli"] += t_nli_raw

        # 6. Belief graph construction & propagation
        t0 = time.perf_counter()
        graph = build_graph(claims, pool, nli, c0, cfg, ablate)
        timings["t_graph"] += time.perf_counter() - t0
        total_edges_count += sum(len(c.depends_on) for c in graph.claims)

        gaps = graph.gaps(e.get("theta_gap", 0.6), use_status=use_status)

        if hop >= e.get("max_hops", 2):
            break

        # Fast path check
        if gaps:
            target = graph.weakest(gaps)
        elif graph.min_eff() < e.get("theta_fast", 0.8):
            target = graph.weakest(graph.claims)
        else:
            break  # Fast path exit

        # Query proposal & EIG retrieval
        t0 = time.perf_counter()
        proposed = propose_queries(llm, target.text, tracker, n=3)
        timings["t_query_gen"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        seen = {p["title"] for p in pool}
        best, best_score = [], -1.0
        for qy in (proposed if proposed else [target.text]):
            res = retriever.search(qy, k=k)
            tracker.probe_searches += 1
            novelty = sum(1 for r in res if r["title"] not in seen) / max(len(res), 1)
            score = (1.0 - target.eff) * novelty
            if score > best_score:
                best, best_score = res, score
        tracker.retrievals += 1
        new = [r for r in best if r["title"] not in seen]
        timings["t_subseq_retrieval"] += time.perf_counter() - t0

        if not new:
            break
        pool = (pool + new)[: e.get("pool_cap", 12)]
        hop += 1

    # Final answer confidence
    ans = graph.answer_claim()
    abstain = ans.eff < e.get("theta_abs", 0.4)
    answer = "unknown" if abstain else draft

    timings["t_total_retrieval"] = timings["t_init_retrieval"] + timings["t_subseq_retrieval"]
    timings["t_total_llm"] = timings["t_llm_draft"] + timings["t_sc"] + timings["t_claims"] + timings["t_query_gen"]
    timings["t_wall_end_to_end"] = time.perf_counter() - t_start

    # Passages stats
    unique_titles = len({p["title"] for p in pool})
    total_titles = len(pool)
    dup_rate = (total_titles - unique_titles) / max(total_titles, 1)

    return {
        "id": item["_id"],
        "answer": answer,
        "em": exact_match(answer, item["answer"]),
        "f1": f1_score(answer, item["answer"]),
        "hops": hop,
        "llm_calls": tracker.llm_calls,
        "retrievals": tracker.retrievals,
        "probe_searches": tracker.probe_searches,
        "in_tokens": tracker.in_tokens,
        "out_tokens": tracker.out_tokens,
        "total_tokens": tracker.in_tokens + tracker.out_tokens,
        "nli_pairs": nli_pairs_count,
        "claims_generated": total_claims_count,
        "graph_nodes": len(graph.claims),
        "graph_edges": total_edges_count,
        "passages_retrieved": total_titles,
        "unique_passages": unique_titles,
        "duplicate_passage_rate": dup_rate,
        **timings,
    }


def profile_question_ircot(
    item: Dict[str, Any],
    factory: RetrieverFactory,
    llm: LLM,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute IRCoT with sub-component instrumentation."""
    c = cfg.get("ircot", {})
    max_steps = c.get("max_steps", 4)
    pool_cap = c.get("pool_cap", 12)
    k = cfg["retrieval"].get("k", 3)
    q = item["question"]

    tracker = Tracker()
    timings = {}
    t_start = time.perf_counter()

    # Initial retrieval
    t0 = time.perf_counter()
    retriever = factory.for_item(item)
    pool = retriever.search(q, k=k)
    timings["t_init_retrieval"] = time.perf_counter() - t0
    tracker.retrievals += 1

    timings["t_subseq_retrieval"] = 0.0
    timings["t_reasoning_steps"] = 0.0
    steps = []

    for _ in range(max_steps):
        t0 = time.perf_counter()
        from src.systems.ircot import next_step
        s = next_step(llm, q, pool, steps, tracker)
        timings["t_reasoning_steps"] += time.perf_counter() - t0
        steps.append(s)

        if "so the answer is" in s.lower() or not s:
            break

        t0 = time.perf_counter()
        seen = {p["title"] for p in pool}
        new = [p for p in retriever.search(s, k=k) if p["title"] not in seen]
        tracker.retrievals += 1
        pool = (pool + new)[:pool_cap]
        timings["t_subseq_retrieval"] += time.perf_counter() - t0

    # Final reader answer
    t0 = time.perf_counter()
    answer = llm.chat(answer_messages(q, pool), tracker, temperature=0.0)
    timings["t_final_reader"] = time.perf_counter() - t0

    timings["t_total_retrieval"] = timings["t_init_retrieval"] + timings["t_subseq_retrieval"]
    timings["t_total_llm"] = timings["t_reasoning_steps"] + timings["t_final_reader"]
    timings["t_wall_end_to_end"] = time.perf_counter() - t_start

    unique_titles = len({p["title"] for p in pool})
    total_titles = len(pool)
    dup_rate = (total_titles - unique_titles) / max(total_titles, 1)

    return {
        "id": item["_id"],
        "answer": answer,
        "em": exact_match(answer, item["answer"]),
        "f1": f1_score(answer, item["answer"]),
        "hops": len(steps),
        "llm_calls": tracker.llm_calls,
        "retrievals": tracker.retrievals,
        "in_tokens": tracker.in_tokens,
        "out_tokens": tracker.out_tokens,
        "total_tokens": tracker.in_tokens + tracker.out_tokens,
        "passages_retrieved": total_titles,
        "unique_passages": unique_titles,
        "duplicate_passage_rate": dup_rate,
        **timings,
    }


def main():
    out_dir = "research/methodology_audit"
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = "data/manifests/dev.jsonl"

    print("=== [Stage 3 Latency & Efficiency Profiling Benchmark] ===")
    print(f"Loading Dev-100 questions from {manifest_path}...")
    with open(manifest_path, "r", encoding="utf-8") as f:
        dev_items = [json.loads(line) for line in f if line.strip()]

    # Load frozen configuration
    cfg = load_cfg("configs/base.yaml", "configs/final.yaml")
    cfg.setdefault("run", {})["cache"] = True
    cfg["run"]["cache_namespace"] = "dev_v1"

    # Cold start timing
    t_cold_0 = time.perf_counter()
    print("Loading models (Cold-Start)...")
    embedder = Embedder(cfg["retrieval"]["embedder"])
    factory = RetrieverFactory(cfg["retrieval"]["setting"], embedder)
    llm = LLM(cfg, use_cache=True, namespace="dev_v1")
    nli = NLI(cfg["nli"]["model"], window=cfg["nli"]["window"])
    cold_start_time = time.perf_counter() - t_cold_0
    print(f"Cold-start model initialization time: {cold_start_time:.3f} seconds\n")

    # Profile E-RAG Active across Dev-100
    print("Running granular profiling for E-RAG Active on Dev-100...")
    erag_records = []
    for item in dev_items:
        rec = profile_question_erag(item, factory, llm, nli, cfg)
        erag_records.append(rec)
    df_erag = pd.DataFrame(erag_records)

    # Profile IRCoT across Dev-100
    print("Running granular profiling for IRCoT on Dev-100...")
    ircot_records = []
    for item in dev_items:
        rec = profile_question_ircot(item, factory, llm, cfg)
        ircot_records.append(rec)
    df_ircot = pd.DataFrame(ircot_records)

    # Compute Statistical Summaries
    timing_cols_erag = [
        "t_wall_end_to_end", "t_init_retrieval", "t_subseq_retrieval", "t_total_retrieval",
        "t_llm_draft", "t_sc", "t_claims", "t_nli", "t_graph", "t_query_gen", "t_total_llm",
    ]
    
    summary_rows = []
    for col in timing_cols_erag:
        vals = df_erag[col].values * 1000.0  # convert to milliseconds
        summary_rows.append({
            "Component / Metric": col,
            "Mean (ms)": np.mean(vals),
            "Median (ms)": np.median(vals),
            "Std (ms)": np.std(vals),
            "Min (ms)": np.min(vals),
            "P75 (ms)": np.quantile(vals, 0.75),
            "P90 (ms)": np.quantile(vals, 0.90),
            "P95 (ms)": np.quantile(vals, 0.95),
            "Max (ms)": np.max(vals),
        })
    df_erag_summary = pd.DataFrame(summary_rows)

    timing_cols_ircot = [
        "t_wall_end_to_end", "t_init_retrieval", "t_subseq_retrieval", "t_total_retrieval",
        "t_reasoning_steps", "t_final_reader", "t_total_llm",
    ]
    ircot_summary_rows = []
    for col in timing_cols_ircot:
        vals = df_ircot[col].values * 1000.0
        ircot_summary_rows.append({
            "Component / Metric": col,
            "Mean (ms)": np.mean(vals),
            "Median (ms)": np.median(vals),
            "Std (ms)": np.std(vals),
            "Min (ms)": np.min(vals),
            "P75 (ms)": np.quantile(vals, 0.75),
            "P90 (ms)": np.quantile(vals, 0.90),
            "P95 (ms)": np.quantile(vals, 0.95),
            "Max (ms)": np.max(vals),
        })
    df_ircot_summary = pd.DataFrame(ircot_summary_rows)

    # Save detailed profiling datasets
    erag_csv = os.path.join(out_dir, "validated_plot_data", "dev100_erag_active_profiling.csv")
    ircot_csv = os.path.join(out_dir, "validated_plot_data", "dev100_ircot_profiling.csv")
    df_erag.to_csv(erag_csv, index=False)
    df_ircot.to_csv(ircot_csv, index=False)
    print(f"Saved profiling datasets:\n  {erag_csv}\n  {ircot_csv}")

    # Display profiling tables
    print("\n=== E-RAG Active Micro-Component Latency Breakdown (milliseconds) ===")
    print(df_erag_summary.to_string(index=False))

    print("\n=== IRCoT Micro-Component Latency Breakdown (milliseconds) ===")
    print(df_ircot_summary.to_string(index=False))

    # Additional Efficiency Metrics
    print("\n=== Additional Efficiency Metrics (Mean per Question on Dev-100) ===")
    print(f"E-RAG Active NLI Pairs Evaluated: {df_erag['nli_pairs'].mean():.2f}")
    print(f"E-RAG Active Claims Generated:    {df_erag['claims_generated'].mean():.2f}")
    print(f"E-RAG Active Graph Nodes / Edges: {df_erag['graph_nodes'].mean():.2f} / {df_erag['graph_edges'].mean():.2f}")
    print(f"E-RAG Active LLM Invocations:     {df_erag['llm_calls'].mean():.2f}")
    print(f"E-RAG Active Total Tokens:        {df_erag['total_tokens'].mean():.1f}")
    print(f"IRCoT Total Tokens:               {df_ircot['total_tokens'].mean():.1f}")
    print(f"Cold-Start Initialization:        {cold_start_time:.3f} s")


if __name__ == "__main__":
    main()
