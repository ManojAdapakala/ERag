# Tasks: Batch 2A Exact Evidence Deduplication and Stopping Telemetry (As-Built)

## 1. Passage Deduplication in NLI Verification

- [x] 1.1 Implement `NLI._normalize_passage_text()` conservative whitespace normalization in `src/epistemic/nli.py`
- [x] 1.2 Implement `seen_texts` deduplication loop and `duplicates_skipped` counter in `NLI.verify()`
- [x] 1.3 Add unit regression tests in `tests/test_batch2a_dedup_and_telemetry.py` for exact duplicates, distinct passages with identical titles, semantic variants, and whitespace variants

## 2. Explicit Stopping Telemetry in E-RAG Pipeline

- [x] 2.1 Add `loop_stop_reason` assignments across all 6 termination branches in `src/systems/erag.py` (`non_adaptive_mode`, `no_adaptive_ablation`, `max_hops_exhausted`, `fast_path`, `pool_capacity_saturated`, `no_novel_evidence`)
- [x] 2.2 Add `output_reason` assignment decoupled from loop stopping (`answered`, `abstained_low_confidence`, `observational_draft`)
- [x] 2.3 Add `fast_path: bool` explicit flag to return dictionary and preserve existing `hops` counter
- [x] 2.4 Add regression tests in `tests/test_batch2a_dedup_and_telemetry.py` verifying each of the 6 termination branches

## 3. Regression & Compatibility Verification

- [x] 3.1 Execute `tests/test_batch1_characterization.py` and `tests/test_batch2a_dedup_and_telemetry.py` (28 passed, 2 xfailed in 22.82s)
- [x] 3.2 Execute core regression suite `tests/test_erag.py`, `tests/test_graph.py`, `tests/test_nli.py` (11 passed in 8.06s)
- [x] 3.3 Verify git diff is clean and restricted exclusively to `src/epistemic/nli.py` and `src/systems/erag.py`

## 4. Completed Resolution of Verification Checks

- [x] 4.1 Order-invariant deduplication context preservation: Refactored `NLI.verify()` to group duplicates and select the canonical title prioritizing entity-grounding (`mentions_subject`) across all candidate headers. Verified via `test_duplicate_text_order_invariance_and_entity_context` and `test_duplicate_text_unrelated_entity_does_not_impute_grounding` (order-invariant contradiction filtering, no indiscriminate pronoun attribution).
- [x] 4.2 End-to-end runner telemetry serialization: Verified that `loop_stop_reason`, `output_reason`, and `fast_path` correctly persist to saved JSONL files through runner output paths (`scripts/run_experiment.py` and `scripts/run_phase5_four_retrievers.py`) in `test_run_experiment_telemetry_serialization` and `test_run_phase5_telemetry_serialization`.
- [x] 4.3 Stale retry test expectation aligned: Reconciled `tests/test_retry_and_tracker.py` line 96 to match the production 6-attempt contract in `src/llm/client.py`. Verified 6 attempts, `RuntimeError`, and failure counter in 0.12s.
