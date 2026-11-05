# AI Assistance and Audit Log

This log records every significant automated design decision, prompt instruction, and architectural file modification performed during the E-RAG research implementation.

---

### Entry 1 — 2026-11-05
- **Stage:** Stage 0 (Read-Only Repository Audit)
- **Objective:** Audit the greenfield repository, local system environment, package compatibility, and investigate baseline discrepancy ($0.54$ vs $0.56$) and Phase 2 drop ($0.28$).
- **Action:**
  - Read full 55-page primary specification `E-RAG Implementation Guide: End to End` (`C:\Users\hp\Downloads\E-RAG Implementation Guide End to End.pdf`).
  - Read research paper manuscript `E_RAG_Detailed_Research_Paper.pdf`.
  - Explored all drives, previous conversation logs, and environment variables.
  - Reconciled mathematical cause of $0.54$ vs $0.56$ (1 question difference on 50-item debug split) and Phase 2 collapse (14 questions lost to naive `"unknown"` abstention on intermediate claims).
  - Authored comprehensive audit artifact: `research/audit.md`.
- **Approval:** User approved proceeding.

---

### Entry 2 — 2026-11-05
- **Stage:** Stage 1 (Project Foundation)
- **Objective:** Setup Python virtual environment `.venv`, pinned requirements, configuration loader, Groq LLM client with SQLite caching and retry backoff, Tracker dataclass, and cache tests.
- **Action:**
  - Initialized git repository.
  - Created `.gitignore` and `.env.example`.
  - Created virtual environment `.venv` with Python 3.14.6.
  - Installed core packages (`groq`, `pyyaml`, `pytest`, `python-dotenv`, `tqdm`, `scikit-learn`, `scipy`, `numpy`, `matplotlib`, `torch`, `sentence-transformers`, `faiss-cpu`, `transformers`, `mlflow`).
  - Locked dependencies in `requirements.lock.txt`.
  - Implemented `configs/base.yaml`.
  - Implemented `src/config.py` with recursive `deep_update` and `load_cfg`.
  - Implemented `src/llm/prompts.py` with standard context block formatting and QA prompts.
  - Implemented `src/llm/client.py` with SQLite caching, exponential backoff, SHA-256 keying over `(ns, model, messages, temp, max_tokens, reasoning_effort, sample_idx)`, and `Tracker`.
  - Authored `research/decisions.md` documenting decisions D-001 through D-005.
  - Implemented `tests/test_cache.py`, `tests/test_config.py`, `tests/test_retry_and_tracker.py`. All 7 tests passed.
  - Git commit tagged `stage1-v1`.

---

### Entry 3 — 2026-11-05
- **Stage:** Stage 2 (Dataset and Retrieval)
- **Objective:** Download HotpotQA, generate stratified manifests (`debug-50`, `dev-100`, `test-400`), implement dense embedder and FAISS retriever (`IndexFlatIP`).
- **Action:**
  - Downloaded official `hotpot_dev_distractor_v1.json` (61 MB, SHA-256: `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8`).
  - Extracted verified 50 debug IDs to `data/raw/debug50_ids.txt`.
  - Implemented `scripts/make_manifests.py` with seeded stratified splitting preserving bridge/comparison distributions.
  - Generated `debug.jsonl` (50), `dev.jsonl` (100), `test.jsonl` (400), and `data/manifests/manifest.json`. Verified mutual disjointness.
  - Implemented `src/retrieval/index.py` (`Embedder`, `paragraphs_from_item`, `build_pooled`, `Retriever`, `RetrieverFactory`).
  - Implemented `tests/test_retrieval.py`. All tests passed.
  - Git commit tagged `manifests-v1`.

---

### Entry 4 — 2026-11-05
- **Stage:** Stage 3 (Evaluation Framework)
- **Objective:** Implement reliable evaluation metrics (EM, F1, Recall, ECE by level, Brier, AUROC) and statistical methods (Bootstrap CI, Paired Bootstrap, exact McNemar, Holm correction).
- **Action:**
  - Implemented `src/eval/metrics.py`. Handled `"unknown"` strictly as 0.0 for EM/F1.
  - Implemented `src/eval/stats.py`.
  - Implemented `tests/test_metrics.py`. All 9 tests passed.
  - Git commit tagged `eval-v1`.

---

### Entry 5 — 2026-11-05
- **Stage:** Stage 4 (Standard RAG Baseline - System A)
- **Objective:** Implement System A (`run_standard`) and closed-book comparator (`run_closed_book`), experiment runner `scripts/run_experiment.py`, and unit tests.
- **Action:**
  - Implemented `src/systems/standard.py`.
  - Implemented `scripts/run_experiment.py` with MLflow tracking, crash recovery/resumption, and JSONL record schema.
  - Implemented `tests/test_standard.py`. All 24 tests in test suite passed.
  - Git commit tagged `baseline-prep-v1`.

---

### Entry 6 — 2026-11-05
- **Stage:** Stage 5 (Epistemic Layer)
- **Objective:** Implement confidence estimation, structured claim decomposition, NLI verification with dynamic label mapping, and belief graph.
- **Action:**
  - Implemented `src/epistemic/confidence.py` (self-consistency sampling for majority and draft agreement).
  - Implemented `src/epistemic/claims.py` (structured JSON extraction with acyclic DAG dependencies and single-claim fallback).
  - Implemented `src/epistemic/nli.py` (sliding sentence windows, dynamic `id2label` discovery, memoized batch scoring).
  - Implemented `src/epistemic/belief_graph.py` (logit evidence updates, weakest-link propagation, evidence gaps detection).
  - Implemented `tests/test_graph.py` and `tests/test_claims.py`. All 33 tests in test suite passed.
  - Git commit tagged `epistemic-v1`.
