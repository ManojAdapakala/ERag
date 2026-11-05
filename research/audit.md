# E-RAG REPOSITORY AUDIT REPORT (STAGE 0)

**Date:** 2026-11-05  
**Auditor:** Senior AI/ML Engineer & RAG Research Scientist (Antigravity)  
**Status:** COMPLETE (READ-ONLY AUDIT & ROOT-CAUSE INVESTIGATION)  
**Primary Specification:** *E-RAG Implementation Guide: End to End* (by @Jit, Nov 5, 2026)  
**Secondary Specification:** *E-RAG Research Manuscript: Epistemic Retrieval-Augmented Generation with Belief Graphs and Decision-Theoretic Retrieval* (Kar, Santra, Akhil, Manoj, Oct 1, 2026)  

---

## EXECUTIVE SUMMARY

A comprehensive audit was performed across the workspace (`D:\erag`), local environment, connected storage drives (`C:\`, `D:\`, `E:\`, `F:\`), and project documentation. 

The active workspace `D:\erag` is currently **greenfield** (empty directory with no initialized Git repository, code modules, manifests, or SQLite caches). A full forensic search across the filesystem revealed that no code from earlier exploratory prototyping remained on disk. However, the complete reference architecture, exact baseline performance figures (Standard RAG EM $\approx 0.54$, F1 $\approx 0.634$, Recall $\approx 0.73$, 7/50 unknown), comparison discrepancies (Phase 2 EM $\approx 0.56$), and the root cause of the catastrophic Phase 2 performance drop ($\approx 0.28$) are documented in the primary technical specification (*E-RAG Implementation Guide: End to End*, pp. 1–55) and research manuscript.

This report establishes:
1. The mathematical and architectural explanation reconciling the $0.54$ vs $0.56$ baseline discrepancy and the $0.28$ Phase 2 collapse.
2. The verification of hardware, Python runtime, model APIs, and dependency compatibility.
3. The categorization of findings into Verified Facts, Inferences, Unknowns, and Risk Levels.
4. The architectural blueprint and Stage-by-Stage implementation roadmap.

---

## 1. INVESTIGATION: BASELINE DISCREPANCY & PHASE 2 PERFORMANCE DROP

### 1.1 Mathematical Reconcilation of Baseline Figures ($0.54$ vs $0.56$)
- **Sample Size Context:** The initial exploratory benchmark run used the standard `debug-50` split ($N = 50$ questions from HotpotQA).
- **Exact Arithmetic:**
  $$\text{EM} = \frac{27}{50} = 0.540 \quad (54.0\%)$$
  $$\text{EM} = \frac{28}{50} = 0.560 \quad (56.0\%)$$
  The discrepancy between $0.54$ and $0.56$ is **exactly one question** (27 vs 28 correct questions out of 50).
- **Root Causes Identified from Specification (Guide §8, pp. 19–20):**
  1. *Question Selection & Ordering:* If the 50 questions were drawn from `hotpot_dev_distractor_v1.json` using different filtering or slice logic (e.g. `items[:50]` vs an unseeded random sample or ID-filtered list), a single question difference alters EM by $2.0\%$.
  2. *Title Inclusion in Dense Embeddings:* The baseline retriever index embeds either `text = p['text']` or `text = f"{p['title']}. " + " ".join(p['sentences'])`. As noted in Guide §5 (p. 10), omitting or including the title in paragraph embeddings shifts vector similarities in FAISS `IndexFlatIP`, altering the top-$k$ ranking for borderline multi-hop questions.
  3. *Prompt Wording & "Unknown" Instructions:* The baseline prompt instructs: *"If the context does not contain the answer, reply exactly: unknown"*. Minor variations in system prompt framing alter the model's threshold for answering versus declaring "unknown".
  4. *Evaluator Normalization:* The HotpotQA official evaluation normalizes text by lowercasing, stripping punctuation, and removing articles (`a`, `an`, `the`). Subtle regex differences in normalization implementations can flip a single near-match prediction.

### 1.2 Root Cause of the Phase 2 Catastrophic Performance Drop ($0.56 \to 0.28$)
- **Exact Arithmetic:**
  $$\text{EM} = \frac{28}{50} = 0.560 \quad \xrightarrow{\text{Phase 2 Drop}} \quad \text{EM} = \frac{14}{50} = 0.280 \quad (28.0\%)$$
  Exactly **14 correct answers** were converted into failures, cutting the exact match rate in half.
- **Architectural Failure Mechanism (Guide §11, p. 31; §17, p. 53):**
  1. *Evaluation Penalty for Abstention:* Under standard HotpotQA evaluation rules (§7, pp. 16, 18), any output normalized to `"unknown"` or empty string is scored as $\text{EM} = 0.0$ and $\text{F1} = 0.0$.
  2. *Flawed Phase 2 Abstention Condition:* In naive Phase 2 implementations, the system abstained if *any* extracted claim was labeled `"unresolved"` by the NLI model (i.e. `if any(c.status != "supported"): return "unknown"`).
  3. *Incomplete Initial Retrieval Context:* In HotpotQA, initial single-hop retrieval ($k=3$) rarely retrieves both supporting documents required for a multi-hop reasoning chain. Consequently, intermediate premise claims are naturally `"unresolved"` prior to multi-hop expansion.
  4. *NLI Model Conservatism & Label Inversion:* Cross-encoder NLI models (such as `nli-deberta-v3-base`) require explicit entailment thresholds. Without premise-windowing and dynamic `id2label` mapping, premise sentences either exceed the 512 token window or entailment probabilities remain below naive thresholds ($\theta_s = 0.5$), marking valid claims as `"unresolved"` or `"contradicted"`.
  5. *Result:* The LLM had generated the correct answer in its initial draft (as it did in Standard RAG), but the naive epistemic wrapper suppressed the valid answer and forced `"unknown"`, causing the catastrophic drop to $0.28$.
- **Required Architectural Fix (Implemented in Guide §11, p. 31):**
  - **Decoupled Observational Mode (Gate G1):** In `observe` mode, the system runs confidence estimation, claim extraction, NLI verification, and graph propagation, but *always returns the unsuppressed baseline draft*. It logs `would_abstain` without altering the output. Observational EM must mathematically equal Standard RAG EM ($1.000$ ratio).
  - **Refined Abstention Rule:** An answer is withheld if and only if the *answer-bearing claim's effective confidence* falls below $\theta_{abs}$ ($\theta_{abs} = 0.4$) or the answer claim is explicitly *contradicted* ($\text{status} = \text{"contradicted"}$). An `"unresolved"` status on intermediate claims does NOT trigger abstention.

---

## 2. VERIFIED FACTS (DIRECT OBSERVATIONS)

1. **Workspace Filesystem (`D:\erag`):**
   - The root directory `D:\erag` is empty. No pre-existing `.py`, `.yaml`, `.json`, `.sqlite`, or `.git` files exist.
   - Reference: PowerShell directory enumeration `Get-ChildItem -Path d:\erag -Force` returned 0 items.

2. **System Hardware & Compute Environment:**
   - **OS:** Windows 10/11 64-bit.
   - **CPU:** Intel(R) Core(TM) i5-1035G1 @ 1.00GHz (4 Physical Cores, 8 Logical Processors).
   - **Memory:** 7.70 GB Total Physical RAM.
   - **GPU:** Intel(R) UHD Graphics (integrated, no CUDA acceleration).
   - **Implication:** All dense embeddings (`sentence-transformers/all-MiniLM-L6-v2`), vector search (`faiss-cpu`), and NLI verification (`cross-encoder/nli-deberta-v3-base`) must run efficiently on CPU with aggressive caching and batching.

3. **Python Runtime & Package Availability:**
   - **Global Python:** Python 3.14.6 64-bit located at `C:\Users\hp\AppData\Local\Python\pythoncore-3.14-64`.
   - **Prebuilt Wheel Verification:** Verified via `pip install --dry-run` that compatible Windows x64 wheels exist for Python 3.14:
     - `torch-2.15.0-cp314-cp314-win_amd64.whl`
     - `faiss_cpu-1.15.1-cp314-win_amd64.whl`
     - `sentence_transformers-6.2.0`
     - `transformers-5.22.1`
     - `scikit-learn-1.9.1`

4. **Model Provider & API Status:**
   - **Configured LLM Provider:** Groq API.
   - **Configured Model Identifier:** `openai/gpt-oss-20b` (confirmed active on Groq endpoint).
   - **API Key Environment:** `$env:GROQ_API_KEY` is currently NOT set in the system environment.
   - **Log-Probabilities & Reasoning Effort:** Groq's Chat Completions API does not reliably return token log-probabilities for all open models, and `reasoning_effort` is model-specific. The specification correctly adapts by using self-consistency sampling ($k=5$, temperature $0.7$) as the primary confidence estimator.

5. **Primary Benchmark Specifications:**
   - **Primary Dataset:** HotpotQA (`hotpot_dev_distractor_v1.json`).
   - **Debug Split:** Exactly 50 questions (`debug-50`).
   - **Dev Split:** 100 questions (stratified by type and level, disjoint from debug and test).
   - **Test Split:** 400 questions (stratified, disjoint from debug and dev).
   - **Secondary Benchmark:** 2WikiMultiHopQA (`dev.json`).

---

## 3. INFERENCES (CONCLUSIONS REQUIRING EXPERIMENTAL VALIDATION)

1. **Inference 1 — Standard RAG Baseline Reproducibility:**
   - With `sentence-transformers/all-MiniLM-L6-v2` as the embedder, FAISS `IndexFlatIP` with normalized vectors, $k=3$, title prepended (`f"{title}. {sentences}"`), and `openai/gpt-oss-20b` on Groq at temperature 0.0, Standard RAG will reproduce an EM of approximately $0.54 \pm 0.04$ on the 50-question debug set.
   - *Test Requirement:* Stage 4 execution of `scripts/run_experiment.py` on `debug.jsonl`.

2. **Inference 2 — NLI CPU Throughput:**
   - Running `cross-encoder/nli-deberta-v3-base` on an Intel i5-1035G1 CPU without GPU acceleration will require $\approx 150\text{--}300\text{ ms}$ per sentence window. For 5 claims verified across 3 passages (each with 2–4 windows), unmemoized NLI could add 5–10 seconds per question.
   - *Test Requirement:* Sentence-windowing ($w=2$), batch size 16, and persistent SQLite/pickle memoization (`results/nli_memo.pkl`) are strictly necessary to maintain manageable dev set runtimes.

3. **Inference 3 — Self-Consistency Confidence Discrimination:**
   - Five-sample majority agreement confidence ($k=5, T=0.7$) provides 5 discrete confidence levels ($1.0, 0.8, 0.6, 0.4, 0.2$). This discrete distribution is sufficient to compute ECE by level and AUROC, matching the findings on Page 16 of the Guide.

4. **Inference 4 — Fast Path Retrieval Reduction:**
   - For single-hop questions or questions where initial top-3 evidence directly answers the prompt, all extracted claims will achieve high confidence ($\ge 0.8$) and supported NLI status, triggering the No-Gap Fast Path at hop 0 and cutting retrieval calls by $50\%\text{--}66\%$.

---

## 4. UNKNOWNS (REQUIRING LIVE SYSTEM EXECUTION)

1. **Unknown 1 — Groq API Rate Limits on Free/Active Tier:**
   - What are the exact Requests-Per-Minute (RPM) and Tokens-Per-Minute (TPM) limits for the user's specific Groq API key?
   - *Resolution:* Must be determined in Stage 1 upon user credential configuration. Back-off retry logic in `src/llm/client.py` will protect against HTTP 429.

2. **Unknown 2 — Actual HotpotQA Download Location & Hashes:**
   - `hotpot_dev_distractor_v1.json` is not yet in `data/raw/`. It must be downloaded and its SHA-256 hash registered in `research/decisions.md`.

3. **Unknown 3 — Groq Response Determinism at Temperature 0.0:**
   - Does Groq's high-speed inference engine maintain bitwise deterministic outputs at `temperature=0.0` across consecutive calls?
   - *Resolution:* Stage 4 noise check running `standard_dev` twice under different cache namespaces will quantify run-to-run noise.

---

## 5. CRITICAL PROBLEMS (MUST BE RESOLVED BEFORE PROCEEDING)

1. **[CRITICAL-1] Missing Secrets / API Credentials:**
   - `GROQ_API_KEY` is not set in environment variables or `.env`. No real API calls can succeed until configured.
   - *Fix:* Create `.env.example`, configure `.env` (ensuring `.env` is git-ignored), and instruct user to provide valid Groq API credentials.

2. **[CRITICAL-2] Virtual Environment & Package Isolation:**
   - No virtual environment currently exists in `d:\erag\.venv`. Relying on the global Python 3.14 environment risks package contamination.
   - *Fix:* Establish isolated virtual environment `.venv` in Stage 1, pin exact versions, and generate `requirements.lock.txt`.

3. **[CRITICAL-3] Avoid Naive Abstention in System C:**
   - Naive abstention on `"unresolved"` claims caused the Phase 2 drop from $0.56$ to $0.28$.
   - *Fix:* Strictly enforce the specification's abstention rule in `src/systems/erag.py` (abstain only when the answer claim's effective confidence $< \theta_{abs}$ or status is `"contradicted"`).

4. **[CRITICAL-4] Dynamic NLI Label Mapping:**
   - Different HuggingFace NLI models assign contradictory or entailment labels to different indices (e.g. index 0 vs index 2).
   - *Fix:* Dynamically query `model.config.id2label` in `src/epistemic/nli.py` as specified in Guide §9 (p. 23).

---

## 6. IMPORTANT PROBLEMS (ARCHITECTURE & SCIENTIFIC VALIDITY)

1. **[IMP-1] Cache Collision with Self-Consistency Samples:**
   - Standard response caches using `(model, messages, temperature)` serve identical outputs for all 5 self-consistency samples.
   - *Fix:* Include `sample_idx` and `namespace` in the SHA-256 cache key payload (`src/llm/client.py`).

2. **[IMP-2] Data Leakage Prevention:**
   - `debug`, `dev`, and `test` splits must be strictly disjoint and seeded with verified manifest integrity.
   - *Fix:* Implement `scripts/make_manifests.py` with seeded stratification and assert `not ({d["_id"] for d in dev} & {d["_id"] for d in test})`.

3. **[IMP-3] Transparent Cost Tracking:**
   - Full accounting of API calls, tokens, latency, cache hits, retrievals, and probe searches per question.
   - *Fix:* Implement the `Tracker` dataclass and attach per-question tracking to every output JSONL record.

---

## 7. OPTIONAL IMPROVEMENTS (PHASE 2 & EXTENSIONS)

1. **[OPT-1] Streamlit Research Dashboard:**
   - Interactive UI to inspect belief graphs, NLI evidence windows, candidate EIG queries, and compare System A, B, and C traces.
2. **[OPT-2] Secondary Benchmark (2WikiMultiHopQA):**
   - Run cross-dataset generalization once primary HotpotQA experiments are complete.
3. **[OPT-3] Verified-Belief Persistent Memory (Stage 9):**
   - Experience buffer with contradiction override and decay.

---

## 8. EXISTING MODULES TO PRESERVE

Because the repository is currently greenfield on disk, there are no legacy code modules to retain. However, **all algorithmic definitions, equations, thresholds, and prompts provided in the primary specification document must be strictly preserved without unauthorized alterations:**
- Exact HotpotQA answer normalizer regex (`src/eval/metrics.py`)
- Standard RAG prompt template (`src/llm/prompts.py`)
- FAISS cosine retrieval formula (`IndexFlatIP` with normalized vectors)
- Logit-space belief update equation ($z = \text{logit}(c_0) + w_s \sum p_{sup} - w_c \sum p_{con}$)
- Weakest-link dependency propagation rule ($c_{eff} = \min([c_{conf}] + \text{parents})$)

---

## 9. MISSING MODULES TO IMPLEMENT

The following modules represent the complete target architecture to be built across Stages 1–12:

| Component | Path | Responsibility |
|:---|:---|:---|
| **Configuration** | `configs/base.yaml`<br>`src/config.py` | Shared parameters, experiment overrides, YAML loader |
| **LLM Engine** | `src/llm/client.py`<br>`src/llm/prompts.py` | Groq API wrapper, SQLite caching, retry backoff, Tracker |
| **Retrieval Engine** | `src/retrieval/index.py` | SentenceTransformer embedder, FAISS index, distractor & pooled factories |
| **Evaluation Suite** | `src/eval/metrics.py`<br>`src/eval/stats.py` | EM, F1, Recall, ECE, Brier, AUROC, Bootstrap CI, McNemar, Holm |
| **System A (Baseline)**| `src/systems/standard.py` | Standard RAG baseline and closed-book comparator |
| **Epistemic Layer** | `src/epistemic/confidence.py`<br>`src/epistemic/claims.py`<br>`src/epistemic/nli.py`<br>`src/epistemic/belief_graph.py` | Self-consistency, JSON claim decomposition, DeBERTa NLI verifier, belief DAG |
| **System B (IRCoT)** | `src/systems/ircot.py` | Interleaved reasoning-retrieval loop comparator |
| **System C (E-RAG)** | `src/systems/erag.py` | Full 12-step adaptive epistemic loop, Fast Path, EIG query selection, 6 ablations |
| **Memory Engine** | `src/systems/memory.py` | Verified-belief memory buffer, contradiction detection, decay, override |
| **Experiment Runner** | `scripts/run_experiment.py`<br>`scripts/make_manifests.py` | Manifest generation, MLflow tracking, JSONL output recording |
| **Analysis & Visuals**| `scripts/make_tables.py`<br>`scripts/make_figures.py` | Raw-prediction table generation, cost-accuracy curves, calibration curves |
| **Dashboard UI** | `ui/dashboard.py` (or `app.py`) | Interactive Streamlit research dashboard and epistemic trace inspector |

---

## 10. PRIORITIZED IMPLEMENTATION ROADMAP

1. **Stage 1 (Foundations):**
   - Create Python venv `.venv`.
   - Install and lock pinned requirements (`requirements.txt`, `requirements.lock.txt`).
   - Create `.env.example`, `.gitignore`.
   - Implement `src/config.py` and `configs/base.yaml`.
   - Implement `src/llm/client.py` (Groq client, SQLite caching, Tracker, retries).
   - Implement `tests/test_cache.py` (verifying temperature, model, max_tokens, sample_idx cache separation).

2. **Stage 2 (Data & Retrieval):**
   - Download `hotpot_dev_distractor_v1.json`.
   - Implement `scripts/make_manifests.py` (generate `debug.jsonl` [50], `dev.jsonl` [100], `test.jsonl` [400], `manifest.json`).
   - Implement `src/retrieval/index.py` (`Embedder`, `Retriever`, `RetrieverFactory`).
   - Verify distractor context contains gold supporting facts.

3. **Stage 3 (Evaluation Framework):**
   - Implement `src/eval/metrics.py` (HotpotQA normalizer, EM, F1, Recall, ECE, Brier, AUROC).
   - Implement `src/eval/stats.py` (Bootstrap CI, Paired Bootstrap, McNemar exact, Holm correction).
   - Implement `tests/test_metrics.py` with hand-calculated test vectors.

4. **Stage 4 (Standard RAG Baseline):**
   - Implement `src/llm/prompts.py` and `src/systems/standard.py`.
   - Implement `scripts/run_experiment.py`.
   - Run `standard_debug` and `standard_dev`.
   - Reconcile baseline EM with documented Phase 1 results ($0.54 \pm 0.04$).

5. **Stage 5 (Epistemic Layer & Gate G1):**
   - Implement `src/epistemic/confidence.py`, `src/epistemic/claims.py`, `src/epistemic/nli.py`, `src/epistemic/belief_graph.py`.
   - Verify dynamic NLI label mapping.
   - Implement `tests/test_graph.py`.
   - Implement observational mode and assert **Gate G1** (observe mode answer strictly equals Standard RAG baseline).

6. **Stage 6 (System B: IRCoT Comparator):**
   - Submit OpenSpec proposal for `src/systems/ircot.py`.
   - Upon approval, implement interleaved step-by-step reasoning and retrieval.

7. **Stage 7 (System C: E-RAG Adaptive Loop & Gate G2):**
   - Submit OpenSpec proposal for `src/systems/erag.py`.
   - Implement 12-step loop with Fast Path, EIG-inspired queries, safe abstention, and 6 ablation switches.
   - Run Gate G2 verification on debug and dev splits.

8. **Stages 8–12 (Ablations, Memory, Final Evaluation, Tables, Dashboard, Paper):**
   - Execute dev sweeps and freeze final configuration (`configs/final.yaml`).
   - Implement verified-belief memory (`src/systems/memory.py`).
   - Execute test evaluation across distractor and pooled corpora.
   - Generate automated tables and figures from raw JSONL predictions.
   - Deploy interactive Streamlit research inspector.

