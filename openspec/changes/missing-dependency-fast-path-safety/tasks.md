# Tasks: Missing-Dependency Fast-Path Safety

- [x] 1. Test Suite Creation (Offline Tests First) <!-- id: 1 -->
  - [x] 1.1 Create `tests/test_missing_dependency.py` covering all 8 policy scenarios:
    - Missing prerequisite for a multi-hop answer.
    - Unresolved prerequisite.
    - Unsupported prerequisite ($eff < \theta_{\text{fast}}$).
    - Contradicted prerequisite.
    - Auxiliary unresolved claim that is not answer-critical (allows fast path).
    - Fast path with all required claims supported.
    - Fast path with a missing/absent dependency.
    - Composite answer with one unsupported component.
  - [x] 1.2 Update `tests/test_batch1_characterization.py`:
    - Remove `@pytest.mark.xfail` from `test_missing_dependency_should_not_fast_path`.
    - Ensure mock components provide appropriate follow-up retrieval responses for bridge queries.

- [x] 2. Implementation <!-- id: 2 -->
  - [x] 2.1 Update `src/epistemic/belief_graph.py`:
    - Support tracking `missing_deps` when dependencies point to invalid/absent indices.
    - Add `critical_gaps()` or update `answer_critical_claims()` evaluation.
  - [x] 2.2 Update `src/epistemic/claims.py`:
    - Track absent dependencies in `depends_on`.
    - Implement minimal deterministic bridge-premise extraction for relational question patterns.
  - [x] 2.3 Update `src/systems/erag.py`:
    - Gate fast-path on answer-critical claims and their dependencies.
    - Allow auxiliary claims to not block fast path when answer-critical claims are fully supported.
    - Select target from critical gaps when fast path is blocked.

- [x] 3. Verification & Regression Testing <!-- id: 3 -->
  - [x] 3.1 Run `tests/test_missing_dependency.py`.
  - [x] 3.2 Run `tests/test_batch1_characterization.py`.
  - [x] 3.3 Run full synchronous offline regression across all test suites.
  - [x] 3.4 Verify 0 failures, 0 strict xfails.
