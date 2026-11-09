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

- [x] 4. Pre-Adaptive-Retrieval Development Smoke Test <!-- id: 4 -->
  - [x] 4.1 Scenario 1: Fast path still works when all answer-critical claims are supported.
  - [x] 4.2 Scenario 2: Missing answer-critical dependencies prevent fast path and trigger retrieval/safe-abstention.
  - [x] 4.3 Scenario 3: Bridge-premise extraction fires only for approved relational patterns.
  - [x] 4.4 Scenario 4: Bridge-premise extraction does not fire on ordinary questions.
  - [x] 4.5 Scenario 5: Bridge claims are prerequisites, not answer-bearing claims.
  - [x] 4.6 Scenario 6: Final answer claim depends on the bridge claim.
  - [x] 4.7 Scenario 7: Contradicted dependencies produce `abstained_contradiction_veto`.
  - [x] 4.8 Scenario 8: Auxiliary unresolved claims do not unnecessarily block fast path.
  - [x] 4.9 Scenario 9: JSONL, provenance, retrieval metadata, claim records, confidence, runner output remain valid.
  - [x] 4.10 Scenario 10: No excessive retrieval on ordinary supported questions or incorrect abstention.
