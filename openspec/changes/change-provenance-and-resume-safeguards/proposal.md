# Proposal: Experiment Provenance and Resume Safeguards

## Why
Audits 1 and 2 established that while benchmark evaluation datasets (e.g. HotpotQA Test-400) contain exact ID alignment, the experiment execution pipeline in primary runners lacks cryptographically verifiable provenance and strict resume safety:
1. **Missing Execution Provenance:** JSONL output files record question-level predictions but lack verified run-level execution metadata (exact git commit, tracked source diff, untracked source validation, full configuration digests, and execution provider settings).
2. **Unchecked Resume Contamination:** When resuming an interrupted benchmark run, existing runners read question IDs from the output file and append remaining questions without verifying whether the active code, prompt, model settings, or configuration match the historical records.
3. **Fragile Interruption & Corrupted Tails:** Historical runners do not formally define how interrupted runs, trailing non-newline JSON records, malformed lines, or duplicate IDs are validated on resume.

Establishing strict execution safeguards ensures historical result reproducibility, protects against configuration drift, and prevents accidental data contamination prior to implementing algorithmic changes.

## What Changes
- **Provenance Module (`src/provenance.py`):** Centralize configuration hashing, source tree validation, manifest validation, process locking, safe file append, and sidecar management.
- **Sidecar Metadata File (`<out_path>.meta.json`):** Store immutable generation identity, progress snapshot, and evaluation metadata in distinct top-level sections, preserving full backward compatibility for all downstream JSONL consumers.
- **Fail-Closed Resume Verification:** On resume, verify full SHA-256 digests of resolved configuration, source identity, manifest, corpus, and provider settings. Halt immediately on any mismatch with a detailed diagnostic message. Incompatible appending is strictly prohibited (no mismatch override flag). Evaluation metadata does not participate in generation resume checks.
- **Exact Failed Record Semantics:** Every valid saved record (including `failed=True`) is treated as completed. Resume will never append a duplicate record for an already evaluated question ID.
- **Fail-Closed Malformed Record Policy & Safe Append:** Any unparseable line or corrupted tail causes resume to fail closed without truncation. When resuming a valid file whose final line lacks a trailing newline, exactly one newline character is appended before writing subsequent records, preventing line concatenation.
- **Explicit Source Set Validation:** Inspect execution-relevant directories (`src/`, `scripts/`, `configs/`). Reject untracked code files, require clean tracked state or capture full SHA-256 of `git diff HEAD -- src/ scripts/ configs/`.
- **Explicit Process Locking & Stale Lock Protection:** Guard runs with `<out_path>.lock`. Detect active PIDs vs stale locks. Never silently delete stale locks; require explicit user intervention.
- **Memory Feature Gating:** Explicitly reject `memory.enabled=True` in guarded runners until memory state restoration semantics are formally specified.
- **Primary Runners Covered:** Scope is strictly bounded to `scripts/run_experiment.py` and `scripts/run_phase5_four_retrievers.py`.

## Capabilities

### New Capabilities
- `runner-provenance`: Enforces manifest uniqueness, metadata sidecar persistence, fail-closed resume verification, safe appending, and process locking for primary runners.

## Impact
- **Code:** New module `src/provenance.py`; integration into `scripts/run_experiment.py` and `scripts/run_phase5_four_retrievers.py`.
- **Predictions & Metrics:** Exactly 0.0% change. No prompts, retrieval logic, or inference parameters are modified.
- **Historical Results:** Preserved as-is. Historical result files lacking sidecars fail closed for resume (preventing mutation) and continue to evaluate cleanly in existing analysis scripts.
