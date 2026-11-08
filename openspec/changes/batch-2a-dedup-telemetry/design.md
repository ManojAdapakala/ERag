# Design: Batch 2A Exact Evidence Deduplication and Stopping Telemetry

## Context
During Audit 3, offline inspection of `src/epistemic/nli.py` and `src/systems/erag.py` revealed that evidence updates were vulnerable to artificial amplification through duplicate passage injection, and that reporting metrics conflated zero-hop retrieval exhaustion with Fast Path completion. Batch 2A implements targeted, conservative fixes for both issues without destabilizing core belief graph algorithms.

## Goals / Non-Goals

**Goals:**
- Prevent identical passage text from contributing repeatedly within a single NLI verification call.
- Ensure different passages sharing the same article title remain distinct and available for verification.
- Preserve distinct wording variants of similar evidence (no fuzzy or semantic merging).
- Record exact, unambiguous termination reasons at each branch of the adaptive loop.
- Separate final output/abstention status from loop stopping reasons.
- Maintain full backward compatibility for `hops` count and existing output dictionaries.

**Non-Goals:**
- No modification of contradiction weights ($w_c$), support weights ($w_s$), or threshold formulas.
- No alteration of NLI model architectures or subject-token filtering heuristics.
- No modification of claim decomposition prompts or dependency extraction.
- No change to IRCoT generation budgets or stopping rules.

## Decisions

### 1. Conservative Passage Normalization in NLI Verification
- **Choice:** Normalize passage text by collapsing whitespace runs (`" ".join(raw.split())`) and tracking `seen_texts: Set[str]` inside `NLI.verify()`.
- **Rationale:** Strict whitespace normalization catches formatting variations without risking semantic collision between genuinely distinct facts. Title-only deduplication is rejected because multi-hop retrieval frequently retrieves multiple distinct paragraphs from the same entity article (e.g. *Paris* early history vs. *Paris* modern administration).
- **Diagnostics:** The number of skipped duplicate passages is recorded in `ev["duplicates_skipped"]` for debugging and telemetry.

### 2. Explicit Branch-Level Loop Telemetry
- **Choice:** Assign `stop_reason` directly before each `break` statement in `run_erag`:
  - `non_adaptive_mode`
  - `no_adaptive_ablation`
  - `max_hops_exhausted`
  - `fast_path`
  - `pool_capacity_saturated`
  - `no_novel_evidence`
- **Rationale:** Eliminates the historical flaw of inferring Fast Path from `hops == 0`.
- **Output Policy Decoupling:** Final decision policy assigns `output_reason` (`answered`, `abstained_low_confidence`, `observational_draft`) independently from `loop_stop_reason`.

## Risks / Trade-offs

- **[Subtle Metric Shifts in Stream Evaluation]** → In streaming evaluation with episodic memory, memory passages that duplicate pool passages will no longer artificially double confidence. This improves epistemic calibration accuracy.
- **[Downstream JSON Schema Compatibility]** → Added keys (`loop_stop_reason`, `output_reason`, `fast_path`, `duplicates_skipped`) are strictly additive; all existing keys and metric parsers function without modification.
