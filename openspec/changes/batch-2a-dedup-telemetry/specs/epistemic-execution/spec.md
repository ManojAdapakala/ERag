# epistemic-execution Specification (Delta: Batch 2A)

## Purpose
Defines the behavior of exact passage evidence deduplication within NLI verification and explicit stopping/output telemetry in the E-RAG adaptive retrieval pipeline.

## Requirements

### Requirement: Exact Passage Text Deduplication in Evidence Verification
The NLI evidence verification subsystem SHALL prevent identical passage text from contributing multiple times within a single verification call, while preserving distinct passages from the same article title.

#### Scenario: Duplicate identical passage text
- **GIVEN** a claim and a passage list containing two identical passage texts under the same or different titles
- **WHEN** `NLI.verify(claim, passages)` is executed
- **THEN** it scores the passage text exactly once, returns exactly one entry in `sup` or `con`, and increments `duplicates_skipped` by 1.

#### Scenario: Distinct passages with the same article title
- **GIVEN** a claim and two passages sharing the title `"Paris"` but containing different sentence contents
- **WHEN** `NLI.verify(claim, passages)` is executed
- **THEN** both passages are retained, evaluated, and eligible to contribute to evidence lists, with `duplicates_skipped == 0`.

#### Scenario: Formatting and whitespace variants
- **GIVEN** two passage instances differing only in leading/trailing whitespace or internal spacing runs
- **WHEN** `NLI.verify(claim, passages)` is executed
- **THEN** they are recognized as identical passage text and deduplicated.

### Requirement: Explicit Loop Stopping Telemetry
The adaptive E-RAG execution loop SHALL record an explicit `loop_stop_reason` string identifying the specific branch that terminated iteration.

#### Scenario: Fast Path early exit
- **GIVEN** an active E-RAG execution at Hop 0 where all claims are supported $\ge \theta_{\text{fast}}$ with no gaps
- **WHEN** the loop terminates
- **THEN** `result["loop_stop_reason"]` is set to `"fast_path"`, `result["fast_path"]` is `True`, and `result["hops"]` is 0.

#### Scenario: No novel evidence retrieved
- **GIVEN** an active E-RAG execution where targeted retrieval returns zero new passages
- **WHEN** the loop terminates
- **THEN** `result["loop_stop_reason"]` is set to `"no_novel_evidence"` and `result["fast_path"]` is `False`.

#### Scenario: Maximum hops exhausted
- **GIVEN** an active E-RAG execution reaching `max_hops` iterations
- **WHEN** the loop terminates
- **THEN** `result["loop_stop_reason"]` is set to `"max_hops_exhausted"`.

#### Scenario: Observational mode
- **GIVEN** E-RAG execution with `active = False`
- **WHEN** the loop terminates after Hop 0
- **THEN** `result["loop_stop_reason"]` is set to `"non_adaptive_mode"` and `result["output_reason"]` is set to `"observational_draft"`.

### Requirement: Independent Output Policy Telemetry
The system SHALL record `output_reason` independently from `loop_stop_reason`.

#### Scenario: Low confidence safe abstention
- **GIVEN** an execution that terminated via any loop stop reason, where answer claim effective confidence is below $\theta_{\text{abs}}$
- **WHEN** final output is formatted
- **THEN** `result["output_reason"]` is set to `"abstained_low_confidence"` and `result["abstained"]` is `True`.
