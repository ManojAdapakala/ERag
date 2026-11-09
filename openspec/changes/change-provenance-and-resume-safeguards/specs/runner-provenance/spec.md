# runner-provenance Specification

## Purpose
Specifies manifest validation, immutable execution metadata sidecars, fail-closed resume verification, process locking, and error handling for primary benchmark runners.

## Requirements

### Requirement: Manifest Uniqueness Pre-flight Validation
The runner SHALL validate that all items in an input manifest possess unique, non-empty `_id` values prior to initiating execution.

#### Scenario: Manifest contains duplicate question ID
- **GIVEN** an input dataset containing two items with `_id == "hotpot_42"`
- **WHEN** the runner validates the manifest
- **THEN** it raises `ValueError("Duplicate question ID 'hotpot_42' detected in manifest")` and terminates before making any model or retriever calls.

### Requirement: Linked Metadata Sidecar Creation
The runner SHALL initialize a companion `<out_path>.meta.json` file recording immutable run identity whenever a new prediction file is created.

#### Scenario: New experiment sidecar initialization
- **GIVEN** an empty target output path
- **WHEN** the runner initializes the output file
- **THEN** it atomically writes `<out_path>.meta.json` containing `schema_version: "2.0"`, full 64-character SHA-256 digests of `resolved_config`, `manifest`, `source_diff`, and full 40-character `git_commit_sha`.

### Requirement: Fail-Closed Resume Identity Verification
When resuming an interrupted run into an existing output file, the runner SHALL verify all immutable identity fields in the sidecar against runtime parameters before appending any records.

#### Scenario: Matching resume execution
- **GIVEN** an existing output file with matching immutable sidecar fields
- **WHEN** the runner executes resume
- **THEN** it identifies all completed question IDs from valid lines in the `.jsonl` file and proceeds to execute only remaining questions.

#### Scenario: Mismatched configuration or source halts resume
- **GIVEN** an existing output file whose sidecar `resolved_config_sha256` or `source_identity` differs from runtime
- **WHEN** the runner executes resume
- **THEN** it raises `IncompatibleResumeError` detailing the mismatched fields and terminates before issuing any model calls or appending records.

#### Scenario: Legacy output file without sidecar
- **GIVEN** an existing `.jsonl` file without a companion `.meta.json` file
- **WHEN** the runner executes resume
- **THEN** it raises `LegacyOutputError` and refuses to resume or append to the legacy file.

### Requirement: Failed Record Invariance
The runner SHALL treat every valid record in an existing output file, including records where `failed == True`, as completed.

#### Scenario: Question with failed status is skipped
- **GIVEN** an output file containing a valid record with `"id": "q_100"` and `"failed": True`
- **WHEN** the runner resumes execution
- **THEN** `"q_100"` is added to `done_ids` and is not re-executed or duplicated.

### Requirement: Damaged Tail & Corruption Detection
The runner SHALL fail closed upon encountering any malformed or corrupted records during resume.

#### Scenario: Trailing truncated JSON line
- **GIVEN** an output file whose final line contains partial or malformed JSON
- **WHEN** the runner attempts to resume
- **THEN** it raises `CorruptedOutputFileError` and refuses to modify or truncate the file.

#### Scenario: Valid final record lacking trailing newline
- **GIVEN** an output file whose final line is valid JSON but lacks a trailing newline character
- **WHEN** the runner attempts to resume
- **THEN** it parses the final record successfully, adds its ID to `done_ids`, ensures a single newline is appended to the file prior to writing subsequent records without modifying or truncating prior records, and continues.

### Requirement: Decoupled Evaluation Metadata
The sidecar SHALL segregate mutable evaluation metadata from immutable generation identity.

#### Scenario: Downstream evaluation update does not invalidate resume
- **GIVEN** an existing `<out_path>.meta.json` containing an `evaluation` section with metric scores
- **WHEN** the generation runner verifies resume compatibility
- **THEN** differences in `evaluation` metadata do NOT cause `IncompatibleResumeError` and do not invalidate generation identity.

### Requirement: Process Locking & Stale Lock Safety
The runner SHALL acquire `<out_path>.lock` during execution and prevent concurrent or stale conflicts.

#### Scenario: Concurrent process conflict
- **GIVEN** an existing `<out_path>.lock` containing an active PID
- **WHEN** another runner instance starts targeting the same output
- **THEN** it raises `ActiveProcessLockError` and exits immediately.

#### Scenario: Stale lock detection
- **GIVEN** an existing `<out_path>.lock` containing a dead PID from an interrupted process
- **WHEN** a runner instance starts targeting the output
- **THEN** it raises `StaleLockDetectedError` and refuses to silently remove the lock without explicit user flag `--clear-stale-lock`.

### Requirement: Memory Feature Gating
The runner SHALL reject runs configured with `memory.enabled == True`.

#### Scenario: Memory enabled in guarded runner
- **GIVEN** a configuration where `cfg["memory"]["enabled"] is True`
- **WHEN** the runner validates execution parameters
- **THEN** it raises `NotImplementedError` stating memory state provenance is deferred.
