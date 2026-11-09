# Spec: Fast-Path Safety Gate and Missing-Dependency Handling

## Requirements

### Requirement 1: Fast-Path Answer-Critical Scope
The fast path early termination condition at Hop 0 shall evaluate whether all answer-critical claims (the union of all answer-bearing claims and their transitive upstream dependency ancestors) are supported with effective confidence $eff \ge \theta_{\text{fast}}$.

Auxiliary claims that are not answer-critical shall not prevent the system from taking the fast path when all answer-critical claims are supported.

### Requirement 2: Missing and Absent Dependency Handling
When an answer-critical claim depends on an absent or unresolved prerequisite:
1. Fast path shall be blocked (`fast_path = False`).
2. Retrieval shall proceed targeting the ungrounded or weakest critical claim.
3. If retrieval budget is exhausted without evidence resolving the prerequisite, `loop_stop_reason` shall be `"max_hops_exhausted"` (or `"no_novel_evidence"` if no new passages are found) and `output_reason` shall be `"abstained_low_confidence"`.

### Requirement 3: Deterministic Relational Bridge Premise
When a question introduces an ungrounded entity/relation (e.g. "Where was the director of Inception born?") not mentioned in the draft answer or initial pool:
1. Claim decomposition shall extract the intermediate bridge claim (e.g. "Christopher Nolan directed Inception") as a prerequisite dependency.
2. The answer claim shall declare dependency on this bridge claim.
3. Because the bridge claim is unverified at Hop 0, fast-path early exit is blocked and retrieval targeting the bridge premise is initiated.

### Requirement 4: Composite Answer Support
When an answer consists of multiple answer-bearing claims (composite answer):
1. All answer-bearing claims are answer-critical.
2. If any answer-bearing claim is unsupported or falls below $\theta_{\text{fast}}$, fast path is blocked.
3. Final answer selection (`graph.answer_claim()`) selects the weakest answer claim, ensuring the system abstains if any component claim remains unsupported.
