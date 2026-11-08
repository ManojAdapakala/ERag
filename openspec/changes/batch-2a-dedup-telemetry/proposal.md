# Proposal: Batch 2A Exact Evidence Deduplication and Stopping Telemetry (As-Built)

## Why
Audits 1–3 and initial Batch 1 characterization tests revealed two concrete implementation defects in the E-RAG execution pipeline:
1. **Passage Evidence Multiplication:** In `src/epistemic/nli.py`, passages passed to `NLI.verify()` (e.g. from pool + memory overlap or duplicate retrieval) were scored and appended to evidence lists without deduplication. When identical passage text appeared twice, its logit contribution was doubled, distorting belief graph calibration.
2. **Conflated Telemetry:** Historical benchmarks and reports conflated `hops == 0` with the Fast Path early-exit mechanism. In reality, `hops == 0` also occurs when targeted retrieval finds no novel documents, when pool capacity saturates, or when running in non-adaptive mode. Furthermore, final abstention decisions were not tracked separately from loop termination reasons.

This change records the as-built retrospective specification and regression testing for these two surgical fixes.

## What Changes
- **Conservative Exact Passage Deduplication:**
  - Added `NLI._normalize_passage_text()` to normalize passage sentences/text strictly (collapsing internal whitespace runs, trimming, preserving casing and punctuation).
  - Deduplicated passages by normalized text within each `NLI.verify()` call so that identical passage text contributes at most once.
  - Preserved distinct passages from the same article title and preserved source references (`duplicates_skipped` counter added to verification output).
  - Maintained all existing support/contradiction thresholds and weights.
- **Explicit Stopping Telemetry:**
  - Added `loop_stop_reason` to `run_erag` return record, recording the exact branch where iterative execution terminated:
    - `"fast_path"`: All claims supported $\ge \theta_{\text{fast}}$ with zero gaps.
    - `"no_novel_evidence"`: Targeted retrieval returned zero novel documents.
    - `"pool_capacity_saturated"`: Pre-check detected pool size $\ge \text{pool\_cap}$.
    - `"max_hops_exhausted"`: Iteration counter reached `max_hops`.
    - `"non_adaptive_mode"`: Running in observational mode (`active=False`).
    - `"no_adaptive_ablation"`: Ablation active for adaptive retrieval.
  - Added `output_reason` to record final decision policy separately from loop termination:
    - `"answered"`: High-confidence non-abstained answer.
    - `"abstained_low_confidence"`: Answer claim effective confidence dropped below $\theta_{\text{abs}}$.
    - `"observational_draft"`: Draft answer produced in observe mode.
  - Added `fast_path: bool` flag explicitly derived from `loop_stop_reason == "fast_path"`.
  - Preserved existing `hops` integer semantics and schema backward compatibility.

## Capabilities

### Modified Capabilities
- `epistemic-execution`: Adds exact passage text deduplication in evidence verification and explicit loop/output stopping telemetry to the adaptive execution loop.

## Impact
- **Code:** Surgical modifications to `src/epistemic/nli.py` and `src/systems/erag.py`.
- **Predictions:** Affects predictions only in cases where duplicate passage text existed in `pool + mem_passages` (eliminates artificial evidence doubling). Zero change on disjoint passages.
- **Historical Results:** Preserved as-is. Schema additions are purely additive and backward compatible.
