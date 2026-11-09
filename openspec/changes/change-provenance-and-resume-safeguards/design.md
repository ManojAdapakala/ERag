# Design: Experiment Provenance and Resume Safeguards

## Context
Scientific reproducibility requires unequivocal proof of the exact code, configuration, prompt, and provider parameters used to generate any benchmark artifact. This design specifies a strict provenance and resume verification architecture for the two primary runners: `scripts/run_experiment.py` and `scripts/run_phase5_four_retrievers.py`.

## Goals / Non-Goals

**Goals:**
- Maintain 100% backward compatibility for all 12 existing downstream analysis and metric scripts by utilizing a linked metadata sidecar (`<out_path>.meta.json`).
- Verify full 64-character SHA-256 digests of all execution-relevant artifacts before permitting a run to execute or resume.
- Enforce fail-closed resumption: abort before modifying any output file or issuing any LLM/provider calls on any identity discrepancy.
- Define unambiguous semantics for failed records, damaged/truncated tails, process locking, and stale locks.
- Gate memory-enabled execution until memory state provenance is formally solved.

**Non-Goals:**
- No modification of LLM cache schemas or eviction of historical cache rows.
- No historical record re-writing, in-place editing, or attempt-history tracking.
- No modifications to legacy archived runners (`run_phase2_experiments.py`, `run_phase3_evaluation.py`, `run_phase4_ircot_bm25.py`, `run_ablations.py`, `run_memory_suite.py`).
- No modifications to historical result files in `results/`.

---

## Decisions

### 1. File Architecture & Consistency Protocol
- **Predictions File (`<out_path>.jsonl`):** Pure question-level records. JSONL is the authoritative source of truth for completed question IDs.
- **Sidecar File (`<out_path>.meta.json`):**
  - **Top-Level Sections:**
    - `schema_version`: `"2.0"`
    - `run_id`: Unique run identifier string.
    - `created_at`: ISO 8601 timestamp.
    - `status`: `"running"` | `"completed"`.
    - `immutable_identity`: Execution generation parameters that MUST match exactly on resume.
    - `progress`: Mutable execution progress tracking.
    - `evaluation`: Mutable evaluation and metric metadata. Distinct from generation identity; evaluation-only updates never invalidate generation resume compatibility.
  - **Immutable Generation Identity (Checked on Resume):**
    - `system`: System name (`"erag"`, `"ircot"`, `"standard"`, `"closed_book"`).
    - `mode`: Execution mode (`"active"`, `"observe"`).
    - `resolved_config_sha256`: Full 64-char SHA-256 of output-relevant configuration.
    - `source_identity`: Full git commit SHA, dirty flag, and diff hash.
    - `manifest_identity`: Full manifest file SHA-256, question ID sequence SHA-256, question count.
    - `retriever_identity`: Setting, type, embedder model, index/corpus hash.
    - `provider_identity`: Model name, temperature, max_tokens, retry limits, cache settings.
  - **Progress Metadata (Snapshot only):**
    - `completed_count`: Integer count of completed questions.
    - `last_updated_at`: Timestamp.
    - `predictions_sha256`: Recorded upon successful run completion (null/omitted while running).
  - **Evaluation Metadata:**
    - Dedicated top-level object reserved for metric scorers and downstream evaluation scripts.
    - Does NOT participate in generation identity comparison during resume verification.
- **Crash Recovery Protocol:**
  - On resume, the runner reads the sidecar to verify immutable identity.
  - Completed questions are discovered by parsing valid JSON records from the `.jsonl` file. A discrepancy between the sidecar's snapshot progress count and the `.jsonl` line count caused by an abrupt crash is safely reconciled by using the valid records in the `.jsonl`.
  - Atomic replacement of the sidecar does NOT make the `.jsonl` and sidecar pair atomic; instead, the `.jsonl` line content remains the authoritative source for completed IDs, and the sidecar progress is an informational snapshot.

### 2. Failed Records Semantics
- Every valid JSON record in the output file, including those with `"failed": True`, is treated as **completed**.
- The runner extracts `done_ids = {row["id"] for row in valid_records}` and skips all of them.
- **Invariant:** The runner will NEVER append a second record for an already-present question ID. Re-evaluating failed questions must occur in a separate linked run.

### 3. Damaged Tails, Corruption Handling & Safe Append Protocol
- The runner reads the output file line-by-line.
- **Valid record without newline & Safe Append Protocol:**
  - If the file ends with a valid JSON string that lacks a trailing newline, it is parsed and preserved as a valid completed record.
  - **Safe Subsequent Append:** Before opening the file for appending subsequent records, the runner inspects the final byte of the non-empty file. If the final byte is not `\n`, it appends exactly one `b"\n"` first. It NEVER truncates, alters, or rewrites existing records.
- **Malformed trailing fragment:** If the final line is incomplete, corrupted, or unparseable JSON (e.g. from an abrupt power loss or SIGKILL mid-write), resume **fails closed**.
  - It raises `CorruptedOutputFileError` explaining the line number and syntax error.
  - It does NOT silently truncate or discard data. Manual inspection or explicit repair is required.
- **Mid-file corruption:** Any unparseable line anywhere in the file raises `CorruptedOutputFileError` and halts resume immediately.

### 4. Source Identity & Untracked Code Policy
- Source tracking is restricted to execution-relevant paths: `src/`, `scripts/`, `configs/`.
- The runner runs `git status --porcelain -- src/ scripts/ configs/`:
  - **Untracked Code:** If any untracked `.py` or `.yaml` file is found under `src/`, `scripts/`, or `configs/`, execution fails closed with `UntrackedCodeError`.
  - **Tracked Modifications:** If tracked files under these directories are dirty, the runner captures:
    - `git_commit`: Full 40-character SHA of `HEAD`.
    - `git_dirty`: `True`.
    - `git_diff_sha256`: Full 64-character SHA-256 of `git diff HEAD -- src/ scripts/ configs/`.
  - Resume requires an exact match on both `git_commit` and `git_diff_sha256`.

### 5. Process Locking & Stale Lock Protocol
- Before reading or writing, the runner acquires `<out_path>.lock`.
- The lockfile contains JSON: `{"pid": <int>, "hostname": "<str>", "started_at": "<iso>"}`.
- If `<out_path>.lock` already exists:
  - Check if process with `pid` is currently active on `hostname`.
  - **Active Process:** Raise `ActiveProcessLockError(f"Process {pid} is actively running.")`.
  - **Dead Process (Stale Lock):** Do NOT silently delete the lock! Raise `StaleLockDetectedError(f"Stale lock from dead process {pid} detected. Manual removal or explicit --clear-stale-lock flag required.")`.

### 6. Execution & Provider Settings Capture
- Configuration hashing captures:
  - Complete nested dictionary under `model`: `name`, `temperature`, `max_tokens`, `reasoning_effort`.
  - Complete nested dictionary under `run`: `cache`, `cache_namespace`, `cache_path`.
  - Complete nested dictionary under `retrieval`: `setting`, `embedder`, `k`, `retriever_type`.
  - Complete nested dictionary under `erag` or `ircot`: all stopping thresholds, sample counts, and weights.
- **Memory Gating:** If `cfg.get("memory", {}).get("enabled") is True`, the runner raises `NotImplementedError("Memory state provenance is deferred and not permitted in guarded runners.")`.

### 7. Removal of Mismatch Appending
- Resumption on mismatched configurations, git commits, manifests, or provider settings is strictly prohibited.
- If any field fails validation, the runner halts before issuing provider requests or modifying files, prints the exact field differences, and instructs the user to provide a new output path.

---

## Acceptance Test Scenarios (`tests/test_provenance_and_resume.py`)

1. **Clean Resume:** Valid output + matching sidecar skips completed records (including `failed=True`) and resumes cleanly.
2. **Config Mismatch:** Mismatched `theta_fast` or `max_tokens` halts resume with descriptive error; 0 API calls, 0 file edits.
3. **Source Mismatch:** Changed commit or modified `git_diff_sha256` halts resume.
4. **Untracked Code Rejection:** An untracked `.py` in `src/` halts execution.
5. **Legacy Rejection:** An existing `.jsonl` without `.meta.json` fails closed.
6. **Duplicate Output ID Rejection:** Output file containing duplicate IDs halts with `CorruptedOutputFileError`.
7. **Malformed Tail Rejection:** Output file ending in broken JSON fails closed without truncation.
8. **Valid Non-Newline Tail:** Output file ending in valid JSON without trailing newline is parsed cleanly.
9. **Active Lock Rejection:** Lockfile with active PID halts execution.
10. **Stale Lock Detection:** Lockfile with dead PID halts with `StaleLockDetectedError` requiring explicit flag.
11. **Memory Gating:** Run with `memory.enabled=True` halts with `NotImplementedError`.
