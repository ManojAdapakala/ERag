# Tasks: Experiment Provenance and Resume Safeguards

## 1. Provenance & Verification Module (`src/provenance.py`)

- [x] 1.1 Implement full 64-char SHA-256 canonical hashing for resolved configurations (captured in `compute_config_hash()`)
- [x] 1.2 Implement source set validation (check untracked files in `src/`, `scripts/`, `configs/`, capture commit and diff SHA-256 in `get_source_identity()`)
- [x] 1.3 Implement manifest pre-flight uniqueness validator (`validate_manifest()`)
- [x] 1.4 Implement atomic sidecar metadata writer and reader (`<out_path>.meta.json`) with decoupled top-level `evaluation` metadata (`SidecarManager`)
- [x] 1.5 Implement fail-closed resume validator checking all immutable identity fields (no mismatch override, `_verify_identity_match()`)
- [x] 1.6 Implement JSONL reader handling valid non-newline tails, failed record inclusion, fail-closed malformed tail detection, and `prepare_output_file_for_append`
- [x] 1.7 Implement process locking with active vs. stale lock detection (`<out_path>.lock`, `RunLock`)

## 2. Runner Integration (Primary Runners Only)

- [x] 2.1 Integrate provenance pre-flight, sidecar lifecycle, safe append, and process lock into `scripts/run_experiment.py`
- [x] 2.2 Add memory feature gating (`memory.enabled=True` raises `NotImplementedError`) in `scripts/run_experiment.py`
- [x] 2.3 Integrate provenance pre-flight, sidecar lifecycle, safe append, and process lock into `scripts/run_phase5_four_retrievers.py`

## 3. Offline Acceptance Testing (`tests/test_provenance_and_resume.py`)

- [x] 3.1 Unit test manifest validation (unique IDs pass; duplicate/empty IDs raise `ValueError`)
- [x] 3.2 Unit test source tree validation (clean tree passes; untracked code in `src/` raises error; diff hash is deterministic)
- [x] 3.3 Unit test sidecar creation and field completeness (all full 64-char hashes present)
- [x] 3.4 Unit test clean resume (skips all valid IDs, including `failed=True`, with zero record duplication)
- [x] 3.5 Unit test fail-closed rejection on config mismatch, git commit mismatch, and legacy file without sidecar
- [x] 3.6 Unit test tail parsing (valid line without newline passes; corrupted trailing fragment raises `CorruptedOutputFileError`)
- [x] 3.7 Unit test process locking (active PID raises `ActiveProcessLockError`; dead PID raises `StaleLockDetectedError`)
- [x] 3.8 Unit test memory gating (`memory.enabled=True` raises `NotImplementedError`)
- [x] 3.9 Unit test safe append protocol (O(1) final byte check ensures trailing newline without rewriting records)
- [x] 3.10 Unit test decoupled evaluation metadata (eval updates do not invalidate generation resume)
- [x] 3.11 Unit test runner lifecycle and mismatch rejection for both primary runners (`run_experiment` & `run_phase5_four_retrievers`)

## 4. Verification & Validation Review

- [x] 4.1 Run full offline test suite `pytest tests/test_provenance_and_resume.py` and confirm 100% pass (27 passed in 28.26s)
- [x] 4.2 Verify zero regression across `test_batch1_characterization.py`, `test_batch2a_dedup_and_telemetry.py`, and core test suites (73 passed, 2 strict xfailed in 18.14s)
- [x] 4.3 Verify zero external API calls, zero model downloads, zero modifications to historical files
