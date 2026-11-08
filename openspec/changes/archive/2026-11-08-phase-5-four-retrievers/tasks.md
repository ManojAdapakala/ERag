# Tasks: Phase 5 Four-Retriever Comparative Benchmark

## 1. Retriever Engine Enhancements

- [x] 1.1 Extend `DenseRetriever` with model-specific query and passage prefix logic in `src/retrieval/index.py`
- [x] 1.2 Update `RetrieverFactory` to register `intfloat/e5-base-v2` and `BAAI/bge-base-en-v1.5` aliases and verify factory dispatch
- [x] 1.3 Add unit test cases in `tests/test_retrieval.py` verifying prefix formatting and factory instantiation, and verify with pytest

## 2. Benchmark Execution Pipeline

- [x] 2.1 Implement `scripts/run_phase5_four_retrievers.py` supporting process locking, resume checkpointing, and 12-cell metric aggregation
- [x] 2.2 Warm up and cache `e5-base-v2` and `bge-base-en-v1.5` embedding models in `.venv`

## 3. Benchmark Evaluation on Test-400

- [x] 3.1 Execute E5-base-v2 evaluations for Standard RAG, IRCoT, and E-RAG-Balanced on Test-400 ($N=400$)
- [x] 3.2 Execute BGE-base-en-v1.5 evaluations for Standard RAG, IRCoT, and E-RAG-Balanced on Test-400 ($N=400$)

## 4. Matrix Aggregation & Scientific Reporting

- [x] 4.1 Aggregate all 12 experimental conditions into `research/retrievers/four_retriever_benchmark.csv` and verify metric completeness
- [x] 4.2 Author comprehensive Phase 5 Completion Report (`research/retrievers/four_retriever_analysis_report.md`) with Headings A through K

## Workflow follow-up

- Validate OpenSpec status via `openspec validate phase-5-four-retrievers`.
- Sync specs and commit deliverables atomically.
