# Design Document: Missing-Dependency Fast-Path Safety

## Policy Matrix

| Case # | Scenario | Fast Path Blocked? | Continue Retrieval? | Expected `loop_stop_reason` | Expected `output_reason` | Minimum Evidence Required |
|:---:|:---|:---:|:---:|:---|:---|:---|
| 1 | **All Required Claims Supported** | No (Fast Path taken) | No | `fast_path` | `answered` | All answer-critical claims and their prerequisite dependencies are present, supported ($eff \ge \theta_{\text{fast}}$), with no contradictions. |
| 2 | **Required Dependency Absent** | Yes | Yes (target missing dep) | `max_hops_exhausted` / `no_novel_evidence` | `abstained_low_confidence` | Missing prerequisite evidence must be retrieved and supported. |
| 3 | **Required Dependency Unresolved** | Yes | Yes (target unresolved dep) | `max_hops_exhausted` / `no_novel_evidence` | `abstained_low_confidence` | Prerequisite evidence must be retrieved with positive entailment. |
| 4 | **Required Dependency Unsupported** ($eff < \theta_{\text{fast}}$) | Yes | Yes (target weakest gap) | `max_hops_exhausted` / `no_novel_evidence` | `abstained_low_confidence` | Sufficient supporting evidence to lift effective confidence $\ge \theta_{\text{fast}}$. |
| 5 | **Required Dependency Contradicted** | Yes | Yes (while budget remains) | `max_hops_exhausted` / `no_novel_evidence` | `abstained_contradiction_veto` | Mitigating strong support ($p \ge \theta_s$) or answer remains vetoed. |
| 6 | **Auxiliary Unresolved Claim** (not answer-critical) | No (Fast Path taken) | No | `fast_path` | `answered` | None (auxiliary claims not needed for answer validity). |
| 7 | **Composite Answer (One Unsupported Component)** | Yes | Yes (target unsupported part) | `max_hops_exhausted` / `no_novel_evidence` | `abstained_low_confidence` | All answer-bearing component claims must achieve support. |
| 8 | **Missing Bridge Prerequisite** (Multi-hop question) | Yes | Yes (target bridge premise) | `max_hops_exhausted` / `no_novel_evidence` (or `fast_path` / answered at later hop) | `abstained_low_confidence` (or `answered` if retrieved) | Evidence establishing the relational bridge connecting question entity to answer entity. |

---

## Technical Specifications

### 1. Answer-Critical Scope in `BeliefGraph`
`BeliefGraph.answer_critical_claims()` returns all answer-bearing claims and their upstream dependency ancestors:
- A claim $C$ is answer-critical iff $C \in \text{critical\_claims}$.
- Any claim $C \notin \text{critical\_claims}$ is an **auxiliary claim**.
- The Fast-Path safety gate evaluates criteria exclusively over `answer_critical_claims()`. Auxiliary claims do NOT block fast path.

### 2. Missing/Absent Dependency Representation
In `src/epistemic/claims.py` and `BeliefGraph`:
- When a claim defines `depends_on` containing indices outside valid topological parents, or indicates absent dependencies, the claim records `missing_deps = True` (or stores absent dependency references).
- A node with `missing_deps = True` cannot have `status = "supported"` and cannot satisfy the fast-path condition.

### 3. Deterministic Bridge-Premise Extraction for the Multi-Hop Pattern
For questions following the relational pattern `"Where was the <relation> of <entity> born?"` (or similar relational descriptions where the question entity is absent from the initial evidence pool, while the draft answer introduces an entity):
- Deterministically extract the bridge premise: e.g. `"<draft_entity> directed <entity>."` (or `"<draft_entity> is the <relation> of <entity>."`).
- Mark the bridge claim with `answer_bearing: False`, `depends_on: []`.
- Mark the final answer claim with `answer_bearing: True`, `depends_on: [bridge_idx]`.
- This ensures the belief graph explicitly represents the bridge dependency. At Hop 0, the bridge premise is unresolved, correctly preventing fast-path exit and targeting retrieval to verify the bridge.

### 4. Fast-Path Condition in `run_erag`
At Hop 0 (and subsequent hops before budget exhaustion):
```python
critical = graph.answer_critical_claims()
critical_supported = bool(critical) and all(
    c.status == "supported" and c.eff >= e.get("theta_fast", 0.8) and not getattr(c, "missing_deps", False)
    for c in critical
)

if "no_conf" in ablate:
    target = graph.answer_claim()
elif not critical_supported:
    # Identify gaps strictly within answer-critical claims first
    critical_gaps = [c for c in critical if c.status != "supported" or c.eff < e.get("theta_gap", 0.6)]
    target = graph.weakest(critical_gaps) if critical_gaps else graph.weakest(critical)
elif gaps:
    # Optional non-critical gaps if any remain
    target = graph.weakest(gaps)
elif "no_fast" in ablate or graph.min_eff() < e.get("theta_fast", 0.8):
    target = graph.weakest(graph.claims)
else:
    stop_reason = "fast_path"
    break
```
Notice:
- If any critical claim is unsupported, has a missing dependency, or has $eff < \theta_{\text{fast}}$, `critical_supported` is `False`. Fast path is blocked, and `target` is selected from `critical_gaps`.
- If all critical claims are supported $\ge \theta_{\text{fast}}$, but an auxiliary claim is unresolved, `critical_supported` is `True`. Fast path CAN exit safely, setting `stop_reason = "fast_path"` and `fast_path = True`.
