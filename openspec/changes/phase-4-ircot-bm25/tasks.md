# Tasks: Phase 4 IRCoT Audit & BM25 Benchmark

## 1. Literature & Methodology Audit
- [x] 1.1 Complete official paper and codebase inspection of Trivedi et al. (ACL 2023)
- [x] 1.2 Author `research/ircot/original_method_audit.md`
- [x] 1.3 Author `research/ircot/original_vs_current_implementation.md`
- [x] 1.4 Compile structured 18-dimension `research/ircot/deviation_register.csv`

## 2. BM25 Lexical Retriever Implementation
- [x] 2.1 Install `rank-bm25` in Python virtual environment
- [x] 2.2 Implement `BM25Retriever` in `src/retrieval/index.py` using `BM25Plus`
- [x] 2.3 Update `RetrieverFactory` to support `retriever_type="bm25"`
- [x] 2.4 Add unit tests for `BM25Retriever` in `tests/test_retrieval.py` and verify all tests pass

## 3. Test-400 BM25 Benchmark Execution
- [x] 3.1 Implement benchmark runner `scripts/run_phase4_ircot_bm25.py`
- [x] 3.2 Evaluate Standard RAG + BM25 on Test-400 (`results/bm25/standard_bm25_test.jsonl`)
- [x] 3.3 Evaluate IRCoT-Controlled + BM25 on Test-400 (`results/bm25/ircot_bm25_test.jsonl`)
- [x] 3.4 Evaluate E-RAG-Balanced + BM25 on Test-400 (`results/bm25/erag_balanced_bm25_test.jsonl`)

## 4. Cross-Retriever Analysis & Reporting
- [x] 4.1 Generate comparative matrix `research/ircot/bm25_vs_dense_benchmark.csv`
- [x] 4.2 Compile formal Phase 4 completion deliverable `research/ircot/ircot_methodology_audit_report.md`
