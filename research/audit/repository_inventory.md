# Repository Inventory & File Census (Phase 0 Audit)

**Date:** 2026-11-07  
**Auditor:** Senior AI Research Engineer & RAG Architect  
**Repository Working Tree:** Clean (`master` branch at commit `ec50c8b`)  
**Scope:** Complete file-by-file audit of `d:\erag`

---

## 1. Directory Tree & Structural Overview

```
D:\erag\
├── configs/
│   ├── base.yaml               # Master default configuration specification
│   └── final.yaml              # Gate G3 frozen configuration (commit 3ec2e9b)
│
├── data/
│   ├── manifests/
│   │   ├── debug.jsonl         # 50-item debug split (HotpotQA)
│   │   ├── dev.jsonl           # 100-item stratified dev split (HotpotQA)
│   │   ├── test.jsonl          # 400-item stratified test split (HotpotQA)
│   │   ├── manifest.json       # SHA-256 hashes and split provenance
│   │   ├── stream.jsonl        # 30-item episodic memory evaluation stream
│   │   └── planted.json        # 5 synthetic false beliefs for contradiction testing
│   └── raw/
│       ├── debug50_ids.txt     # Original 50 HotpotQA question IDs for reproducibility
│       └── hotpot_dev_distractor_v1.json # Raw HotpotQA validation set (61.06 MB)
│
├── src/
│   ├── __init__.py
│   ├── config.py               # YAML configuration loader & deep recursive merger
│   ├── epistemic/
│   │   ├── __init__.py
│   │   ├── belief_graph.py     # DAG belief graph, logit-space updates, weakest-link propagation
│   │   ├── claims.py           # Structured atomic claim extraction & JSON parser
│   │   ├── confidence.py       # Self-consistency consensus confidence estimation
│   │   └── nli.py              # Cross-encoder NLI verifier, sliding window, entity grounding
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py          # HotpotQA exact match, token F1, recall, ECE, Brier, AUROC
│   │   └── stats.py            # Paired bootstrap CI, McNemar exact, Holm-Bonferroni correction
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py           # Groq API client, SQLite cache, exponential retry, Tracker
│   │   └── prompts.py          # System prompts and message builders (standard, closed-book)
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── index.py            # SentenceTransformer Embedder, FAISS IndexFlatIP, RetrieverFactory
│   └── systems/
│       ├── __init__.py
│       ├── standard.py         # System A: Standard RAG and Closed-Book baseline
│       ├── ircot.py            # System B: IRCoT-inspired interleaved reasoning & retrieval
│       ├── erag.py             # System C: E-RAG 12-step epistemic adaptive loop (Active & Observe)
│       └── memory.py           # Verified-belief episodic memory store with contradiction override
│
├── tests/
│   ├── test_cache.py           # SHA-256 cache collision and namespace isolation tests
│   ├── test_claims.py          # Claim decomposition and fallback parsing unit tests
│   ├── test_config.py          # Configuration loading and schema validation tests
│   ├── test_erag.py            # E-RAG Active loop and Gate G1 Observational parity tests
│   ├── test_graph.py           # Belief graph logit updates and topological propagation tests
│   ├── test_ircot.py           # IRCoT iterative stepping and stopping condition tests
│   ├── test_memory.py          # Memory write gating, decay, and contradiction deletion tests
│   ├── test_metrics.py         # HotpotQA evaluation metrics and edge-case tests
│   ├── test_nli.py             # NLI dynamic label mapping and entity-grounded contradiction filter
│   ├── test_retrieval.py       # Embedder normalization and FAISS retrieval tests
│   ├── test_retry_and_tracker.py # Exponential backoff retry and usage tracker unit tests
│   └── test_standard.py        # Standard RAG pipeline execution tests
│
├── scripts/
│   ├── compute_all_sensitivities.py  # Sweeps across 17 hyperparameter configurations
│   ├── eval_erag_vs_ircot.py         # Head-to-head evaluation suite (active vs ircot)
│   ├── eval_memory.py                # Episodic memory stream evaluation runner
│   ├── export_ablation_results.py    # Exports dev ablation metrics to CSV
│   ├── make_figures.py               # Generates calibration, risk-coverage, and Pareto figures
│   ├── make_manifests.py             # Generates seeded stratified debug/dev/test splits
│   ├── make_stream.py                # Generates clustered entity stream for memory testing
│   ├── make_tables.py                # Formats Tables 1, 2, and 3 into markdown and CSV
│   ├── plot_crystal_clear_comparison.py # High-contrast comparative visualization
│   ├── plot_exp1_exp2_active_vs_ircot.py # Plots head-to-head scatter and risk-coverage
│   ├── plot_memory_evaluation.py     # Plots memory stream accuracy and retention
│   ├── profile_efficiency.py         # High-resolution microsecond latency profiling
│   ├── run_ablations.py              # Executes 6 ablation variants on Dev-100
│   ├── run_experiment.py             # Primary experiment runner with MLflow tracking
│   ├── run_memory_suite.py           # Full memory verification test harness
│   ├── run_test_suite.py             # Test-400 benchmark evaluation orchestration
│   ├── verify_observe_parity.py      # Automated Gate G1 parity verification
│   └── weight_grid.py                # Dev hyperparameter calibration grid sweep
│
├── results/
│   ├── cache.sqlite                  # SQLite database storing 11,766 cached LLM completions
│   ├── nli_memo.pkl                  # Memoized DeBERTa-v3 cross-encoder inference pairs
│   ├── closed_book_debug.jsonl       # Closed-Book predictions on Debug-50 (50 items)
│   ├── closed_book_dev.jsonl         # Closed-Book predictions on Dev-100 (100 items)
│   ├── closed_book_test.jsonl        # Closed-Book predictions on Test-400 (400 items)
│   ├── standard_debug.jsonl          # Standard RAG on Debug-50 (50 items)
│   ├── standard_dev.jsonl            # Standard RAG on Dev-100 (100 items)
│   ├── standard_dev_rep.jsonl        # Reproducibility duplicate run on Dev-100 (100 items)
│   ├── standard_test.jsonl           # Standard RAG on Test-400 (400 items)
│   ├── ircot_debug.jsonl             # IRCoT on Debug-50 (50 items)
│   ├── ircot_dev.jsonl               # IRCoT on Dev-100 (100 items)
│   ├── ircot_test.jsonl              # IRCoT on Test-400 (400 items)
│   ├── erag_observe_debug.jsonl      # E-RAG Observe on Debug-50 (50 items)
│   ├── erag_observe_dev.jsonl        # E-RAG Observe on Dev-100 (100 items)
│   ├── erag_observe_test.jsonl       # E-RAG Observe on Test-400 (400 items)
│   ├── erag_active_debug.jsonl       # E-RAG Active on Debug-50 (50 items)
│   ├── erag_active_dev.jsonl         # E-RAG Active on Dev-100 (100 items)
│   ├── erag_active_dev_pooled.jsonl  # E-RAG Active on Dev-100 pooled setting (100 items)
│   ├── erag_active_test.jsonl        # E-RAG Active on Test-400 (400 items)
│   ├── erag_ablate_no_nli_dev.jsonl       # Dev-100 ablation w/o NLI (100 items)
│   ├── erag_ablate_no_conf_dev.jsonl      # Dev-100 ablation w/o Confidence (100 items)
│   ├── erag_ablate_no_prop_dev.jsonl      # Dev-100 ablation w/o Propagation (100 items)
│   ├── erag_ablate_no_adaptive_dev.jsonl  # Dev-100 ablation w/o Adaptive (100 items)
│   ├── erag_ablate_no_fast_dev.jsonl      # Dev-100 ablation w/o Fast Path (100 items)
│   ├── erag_ablate_no_eig_dev.jsonl       # Dev-100 ablation w/o EIG Query (100 items)
│   ├── stream_nomem.jsonl            # Memory stream baseline without memory (30 items)
│   ├── stream_mem.jsonl              # Memory stream with active episodic memory (30 items)
│   ├── stream_planted.jsonl          # Memory stream with 5 planted errors (30 items)
│   └── erag_vs_ircot_study/
│       ├── erag_no_abs_test.jsonl    # E-RAG Uninhibited (theta_abs=0.0) test predictions (400 items)
│       ├── coverage_sweep.csv        # 11-point threshold sweep from theta=0.00 to 0.50
│       └── head_to_head_comparisons.csv # Active vs IRCoT paired metrics
│
├── research/
│   ├── audit.md                      # Historical Stage 0 audit report
│   ├── baseline.md                   # Historical Stage 4 baseline reproduction documentation
│   ├── decisions.md                  # Comprehensive architectural decision record (ADR)
│   ├── method.md                     # Mathematical specification of E-RAG equations
│   ├── final_validation_audit.md     # Authoritative Test-400 validation report (commit 7a3f6fb)
│   ├── openspec_erag.md              # OpenSpec proposal for System C
│   ├── openspec_ircot.md             # OpenSpec proposal for System B
│   ├── openspec_memory.md            # OpenSpec proposal for Episodic Memory
│   ├── ai-assistance-log.md          # Traceability log of automated assistance
│   ├── tables/                       # Benchmark publication tables (CSV and Markdown)
│   ├── figures/                      # Calibration, Risk-Coverage, and Pareto plots
│   ├── methodology_audit/            # Comprehensive 17-figure sensitivity analysis suite
│   └── erag_vs_ircot_study/          # E-RAG Active vs IRCoT head-to-head study artifacts
│
├── requirements.txt                  # Direct project dependencies
├── requirements.lock.txt             # Pinned dependency freeze
├── .env.example                      # Template environment file (GROQ_API_KEY)
├── .env                              # Local credentials (API key configured)
├── .gitignore                        # Git exclusion rules
└── mlflow.db                         # Local SQLite backend for MLflow
```

---

## 2. Core Python Module Inventory

| Module Path | Primary Class / Function | LOC | Role / Purpose |
| :--- | :--- | :---: | :--- |
| `src/config.py` | `load_cfg()`, `_deep_merge()` | 42 | Loads and deeply merges YAML configuration files. |
| `src/llm/client.py` | `LLM`, `Tracker` | 172 | Thread-safe Groq client with SQLite cache, exponential backoff, and per-query tracking. |
| `src/llm/prompts.py` | `answer_messages()`, `context_block()` | 39 | Context-grounded and closed-book QA prompts. |
| `src/retrieval/index.py` | `Embedder`, `Retriever`, `RetrieverFactory` | 101 | SentenceTransformer embedding, FAISS inner-product index, and distractor/pooled factories. |
| `src/epistemic/confidence.py` | `self_consistency()` | 43 | Temperature sampling ($k=5$) computing majority agreement and draft agreement confidence. |
| `src/epistemic/claims.py` | `extract_claims()`, `_parse()` | 82 | Prompt LLM to produce structured JSON atomic claims with backward dependency DAG edges. |
| `src/epistemic/nli.py` | `NLI`, `verify()`, `_extract_subject_entities()` | 200 | Cross-encoder NLI verification with sliding sentence window and entity-grounded contradiction filter. |
| `src/epistemic/belief_graph.py` | `BeliefGraph`, `Claim`, `update()` | 123 | Directed acyclic graph representing claims, logit-space updates, and weakest-link uncertainty propagation. |
| `src/systems/standard.py` | `run_standard()`, `run_closed_book()` | 45 | System A baseline implementations. |
| `src/systems/ircot.py` | `run_ircot()`, `next_step()` | 79 | System B IRCoT-inspired interleaved CoT reasoning and retrieval comparator. |
| `src/systems/erag.py` | `run_erag()`, `targeted_retrieve()` | 215 | System C 12-step epistemic retrieval loop (Active, Observational, Ablations). |
| `src/systems/memory.py` | `BeliefMemory` | 128 | Episodic memory store with verification gating, decay, and contradiction deletion. |
| `src/eval/metrics.py` | `exact_match()`, `f1_score()`, `ece_by_level()` | 101 | HotpotQA answer normalization, exact match, token F1, recall, ECE, Brier, AUROC. |
| `src/eval/stats.py` | `paired_bootstrap()`, `mcnemar_exact()`, `holm()` | 69 | Non-parametric paired bootstrap, exact McNemar test, Holm-Bonferroni correction. |

---

## 3. Test Suite Verification

- **Total Test Files:** 12 files in `tests/`.
- **Total Test Cases:** 45 automated unit and integration tests.
- **Execution Command:** `.venv\Scripts\python -m pytest -q`
- **Result:** **45 passed in 99.74s** (100% pass rate).

