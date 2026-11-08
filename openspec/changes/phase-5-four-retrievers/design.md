# Design: Phase 5 Four-Retriever Comparative Benchmark

## Context
In Phases 0–4, evaluations were conducted using either `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense) or `bm25` (lexical BM25Plus). To validate empirical generality across retrieval paradigms, we expand evaluation to include `intfloat/e5-base-v2` and `BAAI/bge-base-en-v1.5`. Both models use 768-dimensional embeddings and require asymmetric prefixing to achieve optimal retrieval accuracy.

## Goals / Non-Goals

**Goals:**
- Extend `DenseRetriever` to transparently handle asymmetric prefixes based on model identifier.
- Maintain a single, clean factory interface (`RetrieverFactory.create()`).
- Reuse already validated runs from Phase 3 (`all-MiniLM-L6-v2`) and Phase 4 (`bm25`) to avoid redundant API expenditure.
- Execute the 6 new experimental conditions on Test-400 ($N=400$):
  - Standard RAG + E5-base-v2
  - IRCoT-Controlled + E5-base-v2
  - E-RAG-Balanced + E5-base-v2
  - Standard RAG + BGE-base-en-v1.5
  - IRCoT-Controlled + BGE-base-en-v1.5
  - E-RAG-Balanced + BGE-base-en-v1.5
- Compute full metrics for all 12 experimental conditions and produce `research/retrievers/four_retriever_benchmark.csv`.

**Non-Goals:**
- No modification of LLM prompts, agent reasoning logic, or verification thresholds.
- No fine-tuning or adaptation of retrieval embedding models.
- No re-running of already completed and validated BM25 or MiniLM runs.

## Decisions

### 1. Transparent Asymmetric Prefixing in `DenseRetriever`
- **Choice:** Encapsulate prefixing rules inside `DenseRetriever` rather than modifying `EragPipeline`, `IRCoTPipeline`, or `StandardRAG`.
- **Rationale:** Keeps agent logic decoupled from retrieval representations.
- **Rules:**
  - `intfloat/e5-base-v2`: Queries receive `query: `, passages receive `passage: `.
  - `BAAI/bge-base-en-v1.5`: Queries receive `Represent this sentence for searching relevant passages: `, passages receive no prefix.
  - Other models: No prefix.

### 2. Experimental Efficiency & Re-use
- **Choice:** Aggregate existing Phase 3 Test-400 runs (`results/standard_test.jsonl`, `results/ircot_test.jsonl`, `results/cost_opt/erag_balanced_test.jsonl`) and Phase 4 runs (`results/bm25/*_bm25_test.jsonl`).
- **Rationale:** Avoids wasteful LLM token burn while ensuring identical evaluation semantics. Only the 6 missing conditions (E5 and BGE) will be executed.

### 3. Fault-Tolerant Execution Architecture
- Script: `scripts/run_phase5_four_retrievers.py`.
- Windows process locking via `.phase5.lock` to avoid duplicate processes.
- Append-and-resume capability reading existing output JSONL lines.

## Risks / Trade-offs

- **[Model Download & VRAM/RAM Usage]** → Pre-load and cache `e5-base-v2` and `bge-base-en-v1.5` sequentially, ensuring CPU/GPU memory is garbage collected between model switches.
- **[Groq TPM Limits on 2,400 queries]** → Use `rate_limit_pause` with exponential backoff on HTTP 429 and persistent JSONL checkpointing.
