# Proposal: Phase 5 Four-Retriever Comparative Benchmark

## Why
A central scientific challenge in evaluating Retrieval-Augmented Generation (RAG) and iterative multi-hop reasoning systems is proving that empirical gains are general and invariant across retrieval architectures rather than artifacts of a single dense vector embedding space. In Phase 4, we established that E-RAG outperforms Standard RAG and IRCoT under lexical BM25 retrieval. To establish comprehensive empirical validity for peer-reviewed publication (e.g., IEEE Transactions on Knowledge and Data Engineering), we must systematically evaluate all three systems across four heterogeneous, widely-adopted retrieval backbones spanning lexical matching, lightweight dense bi-encoders, asymmetric retrieval models, and state-of-the-art representation models.

## What Changes
- **New Capability:** Introduce multi-retriever support in `src/retrieval/index.py` enabling dynamic selection and evaluation of four distinct retrieval models:
  1. `bm25`: Lexical BM25Plus ($k_1=1.5, b=0.75, \delta=1.0$)
  2. `sentence-transformers/all-MiniLM-L6-v2`: Standard 384-dimensional dense bi-encoder
  3. `intfloat/e5-base-v2`: 768-dimensional dense bi-encoder with asymmetric prefixing (`query: ` and `passage: `)
  4. `BAAI/bge-base-en-v1.5`: 768-dimensional dense representation model with query instruction prefixing (`Represent this sentence for searching relevant passages: `)
- **Asymmetric Prefix Support:** Extend `DenseRetriever` to handle model-specific prefixing rules without contaminating query/passage representation.
- **Benchmark Runner:** Implement an automated benchmark script (`scripts/run_phase5_four_retrievers.py`) with process locking, resume capability, and atomic JSONL output generation.
- **Full $3 \times 4 = 12$ Matrix:** Evaluate Standard RAG, IRCoT, and E-RAG-Balanced across all 4 retrievers on HotpotQA Test-400 ($N=400$), generating `research/retrievers/four_retriever_benchmark.csv`.
- **Publication-Ready Analysis:** Generate formal comparative analysis examining cross-retriever recall transfer, calibration stability, cost-efficiency frontiers, and distractor susceptibility.

## Capabilities

### New Capabilities
- `retrieval`: Extends retrieval engine capabilities to support asymmetric dense bi-encoders (`e5-base-v2`, `bge-base-en-v1.5`) alongside existing `bm25` and `all-MiniLM-L6-v2`, with model-specific query and passage formatting.

### Modified Capabilities
<!-- None: No existing main specs to modify. -->

## Impact
- **Code:** Enhances `src/retrieval/index.py` with prefix formatting and factory model registry.
- **Dependencies:** Uses existing `sentence-transformers`, `torch`, and `rank-bm25` in `.venv`.
- **Artifacts:** Generates Test-400 predictions under `results/phase5/` and comparative matrix in `research/retrievers/`.
- **Baselines:** Zero modifications to frozen baseline predictions (`results/standard_test.jsonl`, `results/ircot_test.jsonl`, `results/cost_opt/erag_balanced_test.jsonl`, `results/bm25/*`).
