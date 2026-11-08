# Change Proposal: Phase 4 IRCoT Methodology Audit & Fair BM25 Baseline

## Why
The primary iterative multi-hop retrieval baseline compared against E-RAG is **IRCoT (Interleaving Retrieval with Chain-of-Thought Reasoning)**, introduced by Trivedi et al. (ACL 2023). 
To ensure scientific defensibility and methodological rigor in an IEEE publication, we must:
1. Conduct a deep audit of the original Trivedi et al. paper and official code (`https://github.com/StonyBrookNLP/ircot`).
2. Clearly delineate the three tiers of IRCoT (Tier 1 core invariant algorithm, Tier 2 open-domain setup, Tier 3 controlled adaptation).
3. Implement a fair sparse lexical baseline (**BM25**) to evaluate whether E-RAG's epistemic retrieval advantage remains consistent across sparse (BM25) and dense (MiniLM) retrieval regimes.

## What Changes
1. **Literature & Method Audit:** Document the original paper mechanics, prompt structures, few-shot demonstrations, stopping criteria, and build a formal 18-dimension deviation register.
2. **BM25 Lexical Retriever:** Implement `BM25Retriever` in `src/retrieval/index.py` using `rank-bm25` (`BM25Plus` with Lucene-compatible non-negative IDF).
3. **Multi-System BM25 Benchmark:** Benchmark Standard RAG + BM25, IRCoT-Controlled + BM25, and E-RAG-Balanced + BM25 on HotpotQA Test-400.
4. **Sparse vs Dense Comparative Matrix:** Produce a cross-retriever performance matrix in `research/ircot/bm25_vs_dense_benchmark.csv`.

## Capabilities
- `BM25Retriever`: In-memory word-level BM25 lexical ranking matching standard BM25 retrieval without external Elasticsearch cluster dependencies.
- Dual-retriever support in `RetrieverFactory(setting, retriever_type="dense"|"bm25")`.
- Comprehensive reproducibility audit for IEEE reviewers.

## Impact
- Establishes that System B is an exact Tier 1 algorithmic reproduction under controlled distractor settings (**IRCoT-Controlled**).
- Verifies whether E-RAG's +12.38% gold paragraph recall advantage over IRCoT persists under lexical keyword matching.
