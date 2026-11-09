# Change Proposal: Missing-Dependency Fast-Path Safety

## Context & Problem Statement
E-RAG features a No-Gap Fast Path (Step 8 & 9) designed to bypass multi-hop retrieval and terminate at Hop 0 when all extracted draft claims have sufficient empirical support ($\ge \theta_{\text{fast}} = 0.80$) and no epistemic gaps are detected.

However, in multi-hop question answering where a draft answer provides a factual conclusion without establishing the required intermediate bridge relation (e.g. Question: *"Where was the director of Inception born?"*, Draft: *"Christopher Nolan was born in London."*), E-RAG currently takes the fast path at Hop 0 if the initial retrieval pool supports the draft's entity birthplace (*Christopher Nolan born in London*), even though the pool contains zero evidence that Christopher Nolan directed *Inception*.

This false fast-path exit occurs because:
1. **Unstated Bridge Premise / Missing Dependency:** The LLM's claim extraction prompt extracts factual claims from the draft and context, but does not extract the unstated bridge relation (*Christopher Nolan directed Inception*) connecting the question entity (*director of Inception*) to the draft entity (*Christopher Nolan*).
2. **Missing Dependency Pruning:** In `src/epistemic/claims.py` and `BeliefGraph.add()`, any dependency reference pointing to an absent index or unstated claim is silently dropped rather than tracked as a missing requirement.
3. **Unsafe Fast-Path Gate:** In `src/systems/erag.py`, the fast path checks only that existing graph nodes have $eff \ge \theta_{\text{fast}}$ and `gaps` is empty. It does not verify that all answer-critical claims and their prerequisite dependencies are present, resolved, and grounded.

This defect is characterized by the strict xfail test:
`tests/test_batch1_characterization.py::test_missing_dependency_should_not_fast_path`.

## Proposed Solution
Implement minimal, bounded missing-dependency and fast-path safety guards:
1. **Fast-Path Safety Gate:** Fast-path early termination at Hop 0 is strictly blocked if any required dependency or answer-critical claim is missing, absent, unresolved, or ungrounded.
2. **Deterministic Bridge-Premise Handling:** For multi-hop questions where the question introduces a relational descriptor/entity (e.g., *"the director of Inception"*) that is ungrounded in the evidence pool and not asserted in the claims, extract the bridge premise (*"Christopher Nolan directed Inception"*) as an upstream dependency of the answer claim (`depends_on: [0]`).
3. **Auxiliary Claim Isolation:** An unresolved or unsupported auxiliary claim that is not answer-critical does NOT block the fast path if all answer-critical claims and their dependency ancestors are fully supported.
4. **Offline Test Coverage:** Provide comprehensive offline tests covering all 8 specified safety scenarios, and remove the strict xfail on `test_missing_dependency_should_not_fast_path`.

## Scope & Non-Goals
- **In Scope:**
  - Bounded fast-path verification in `src/systems/erag.py`.
  - Minimal bridge-premise extraction in `src/epistemic/claims.py` for the tested relational dependency pattern.
  - Tracking absent/unresolved dependencies in `src/epistemic/belief_graph.py`.
  - Offline unit tests in `tests/test_missing_dependency.py` and update of `tests/test_batch1_characterization.py`.
- **Out of Scope (Non-Goals):**
  - No changes to confidence weights ($w_s, w_c$), NLI thresholds, or retrieval ranking.
  - No changes to general LLM prompt templates or IRCoT reasoning.
  - No broad open-world commonsense deduction.
  - No live API calls or model downloads.
