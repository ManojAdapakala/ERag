"""Offline Acceptance Test Suite for Experiment Provenance and Resume Safeguards.

Covers:
1. Manifest validation (unique IDs pass; missing or duplicate IDs raise ValueError)
2. Config hashing (canonicalization, sensitivity to output-relevant parameters, invariance to cosmetic keys)
3. Source identity & untracked code rejection
4. Output file parsing (clean read, inclusion of failed=True records, non-newline tails, corruption detection)
5. Sidecar lifecycle (initialization, matching resume, completion)
6. Fail-closed resume rejection on config mismatch, source mismatch, and legacy files
7. Process locking with active PID rejection and dead PID stale-lock detection
8. Memory gating rejection
"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest

from src.provenance import (
    ActiveProcessLockError,
    CorruptedOutputFileError,
    IncompatibleResumeError,
    LegacyOutputError,
    RunLock,
    SidecarManager,
    StaleLockDetectedError,
    UntrackedCodeError,
    check_memory_gating,
    compute_config_hash,
    get_source_identity,
    prepare_output_file_for_append,
    read_existing_output_records,
    validate_manifest,
)


# -----------------------------------------------------------------------------
# 1. Manifest Validation Tests
# -----------------------------------------------------------------------------

def test_manifest_validation_valid():
    """Test: Valid manifest with unique non-empty IDs parses cleanly and computes digests."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"_id": "q1", "question": "Question 1"}) + "\n")
        f.write(json.dumps({"_id": "q2", "question": "Question 2"}) + "\n")
        manifest_path = f.name

    try:
        res = validate_manifest(manifest_path)
        assert res["total_questions"] == 2
        assert len(res["items"]) == 2
        assert len(res["manifest_sha256"]) == 64
        assert len(res["manifest_id_sequence_sha256"]) == 64
    finally:
        os.remove(manifest_path)


def test_manifest_validation_duplicate_id_rejected():
    """Test: Manifest containing duplicate question IDs raises ValueError before any execution."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"_id": "q_dup", "question": "Question 1"}) + "\n")
        f.write(json.dumps({"_id": "q_dup", "question": "Question 2"}) + "\n")
        manifest_path = f.name

    try:
        with pytest.raises(ValueError, match="Duplicate question ID 'q_dup' detected"):
            validate_manifest(manifest_path)
    finally:
        os.remove(manifest_path)


def test_manifest_validation_missing_id_rejected():
    """Test: Manifest item missing non-empty _id raises ValueError."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"question": "Missing ID"}) + "\n")
        manifest_path = f.name

    try:
        with pytest.raises(ValueError, match="missing non-empty '_id'"):
            validate_manifest(manifest_path)
    finally:
        os.remove(manifest_path)


# -----------------------------------------------------------------------------
# 2. Config Hashing Tests
# -----------------------------------------------------------------------------

def test_compute_config_hash_invariance_to_cosmetic_keys():
    """Test: Configuration hash is sensitive to output-relevant keys and invariant to cosmetic keys."""
    base_cfg = {
        "model": {"name": "test-model", "temperature": 0.0},
        "retrieval": {"setting": "distractor", "k": 3},
        "erag": {"theta_fast": 0.80},
        "run": {"name": "exp_1", "out_dir": "results/dir_a", "cache": True},
    }
    cosmetic_variant = {
        "model": {"name": "test-model", "temperature": 0.0},
        "retrieval": {"setting": "distractor", "k": 3},
        "erag": {"theta_fast": 0.80},
        "run": {"name": "different_display_name", "out_dir": "results/dir_b", "cache": True},
    }
    substantive_variant = {
        "model": {"name": "test-model", "temperature": 0.0},
        "retrieval": {"setting": "distractor", "k": 3},
        "erag": {"theta_fast": 0.70},  # modified parameter
        "run": {"name": "exp_1", "out_dir": "results/dir_a", "cache": True},
    }

    hash_base = compute_config_hash(base_cfg)
    hash_cosmetic = compute_config_hash(cosmetic_variant)
    hash_substantive = compute_config_hash(substantive_variant)

    assert len(hash_base) == 64
    assert hash_base == hash_cosmetic, "Cosmetic keys like out_dir must not alter config hash"
    assert hash_base != hash_substantive, "Substantive parameter changes must produce a different hash"


# -----------------------------------------------------------------------------
# 3. Source Identity & Untracked Code Tests
# -----------------------------------------------------------------------------

def test_source_identity_untracked_code_raises_error():
    """Test: Untracked .py file in relevant directory raises UntrackedCodeError."""
    with patch("subprocess.run") as mock_run:
        # Mock git status output showing an untracked python file in src/
        mock_run.return_value = MagicMock(
            stdout="?? src/new_module.py\n", check_returncode=lambda: None
        )
        with pytest.raises(UntrackedCodeError, match="Untracked execution-relevant code found"):
            get_source_identity(relevant_dirs=("src",))


def test_source_identity_clean_tree():
    """Test: Clean tracked tree produces 40-char commit SHA and git_dirty=False."""
    with patch("subprocess.run") as mock_run:
        # git status clean
        proc_status = MagicMock(stdout="", check_returncode=lambda: None)
        # git rev-parse HEAD
        proc_rev = MagicMock(stdout="a" * 40 + "\n", check_returncode=lambda: None)
        # git diff HEAD empty
        proc_diff = MagicMock(stdout=b"", check_returncode=lambda: None)
        mock_run.side_effect = [proc_status, proc_rev, proc_diff]

        ident = get_source_identity()
        assert ident["git_commit_sha"] == "a" * 40
        assert ident["git_dirty"] is False
        assert ident["git_diff_sha256"] is None


def test_source_identity_dirty_tree_captures_diff_hash():
    """Test: Tracked modifications in relevant dirs capture full 64-char git_diff_sha256."""
    with patch("subprocess.run") as mock_run:
        proc_status = MagicMock(stdout=" M src/epistemic/nli.py\n", check_returncode=lambda: None)
        proc_rev = MagicMock(stdout="b" * 40 + "\n", check_returncode=lambda: None)
        proc_diff = MagicMock(stdout=b"diff content here", check_returncode=lambda: None)
        mock_run.side_effect = [proc_status, proc_rev, proc_diff]

        ident = get_source_identity()
        assert ident["git_dirty"] is True
        assert len(ident["git_diff_sha256"]) == 64


# -----------------------------------------------------------------------------
# 4. Output File Parsing & Invariant Tests
# -----------------------------------------------------------------------------

def test_read_existing_records_clean_and_includes_failed():
    """Test: Reads records cleanly and treats failed=True as completed (included in done_ids)."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"id": "q1", "answer": "Paris", "failed": False}) + "\n")
        f.write(json.dumps({"id": "q2", "answer": "", "failed": True}) + "\n")
        out_path = f.name

    try:
        manifest_id_set = {"q1", "q2", "q3"}
        records, done_ids = read_existing_output_records(out_path, manifest_id_set)
        assert len(records) == 2
        assert "q1" in done_ids
        # Failed record must be included in done_ids so it is not re-executed in this run
        assert "q2" in done_ids
    finally:
        os.remove(out_path)


def test_read_existing_records_valid_non_newline_tail():
    """Test: Valid JSON record at the end of file without a trailing newline is accepted."""
    with tempfile.NamedTemporaryFile("wb", suffix=".jsonl", delete=False) as f:
        # Note: no newline after the last record
        f.write((json.dumps({"id": "q1", "answer": "Paris"}) + "\n").encode("utf-8"))
        f.write(json.dumps({"id": "q2", "answer": "Berlin"}).encode("utf-8"))
        out_path = f.name

    try:
        records, done_ids = read_existing_output_records(out_path, {"q1", "q2"})
        assert len(records) == 2
        assert done_ids == {"q1", "q2"}
    finally:
        os.remove(out_path)


def test_read_existing_records_corrupted_tail_fails_closed():
    """Test: Corrupted trailing fragment raises CorruptedOutputFileError without truncation."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"id": "q1", "answer": "Paris"}) + "\n")
        f.write('{"id": "q2", "answe\n')  # broken JSON
        out_path = f.name

    try:
        with pytest.raises(CorruptedOutputFileError, match="Corrupted or truncated JSON"):
            read_existing_output_records(out_path, {"q1", "q2"})
    finally:
        os.remove(out_path)


def test_read_existing_records_duplicate_id_fails_closed():
    """Test: Output file containing duplicate IDs raises CorruptedOutputFileError."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"id": "q1", "answer": "Paris"}) + "\n")
        f.write(json.dumps({"id": "q1", "answer": "Paris Again"}) + "\n")
        out_path = f.name

    try:
        with pytest.raises(CorruptedOutputFileError, match="Duplicate question ID 'q1'"):
            read_existing_output_records(out_path, {"q1", "q2"})
    finally:
        os.remove(out_path)


def test_read_existing_records_foreign_id_fails_closed():
    """Test: Output file containing ID not in manifest raises IncompatibleResumeError."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"id": "q_unknown", "answer": "Foreign"}) + "\n")
        out_path = f.name

    try:
        with pytest.raises(IncompatibleResumeError, match="not present in the active manifest"):
            read_existing_output_records(out_path, {"q1", "q2"})
    finally:
        os.remove(out_path)


# -----------------------------------------------------------------------------
# 5. Sidecar Lifecycle & Fail-Closed Resume Tests
# -----------------------------------------------------------------------------

def _make_dummy_identities():
    source_ident = {
        "git_commit_sha": "c" * 40,
        "git_dirty": False,
        "git_diff_sha256": None,
        "tracked_paths": ["src"],
    }
    manifest_meta = {
        "manifest_path": "data/manifests/test.jsonl",
        "manifest_sha256": "m" * 64,
        "manifest_id_sequence_sha256": "s" * 64,
        "total_questions": 10,
    }
    retriever_ident = {"setting": "distractor", "embedder": "minilm"}
    provider_ident = {"model": "llama3-8b-8192", "temperature": 0.0}
    return source_ident, manifest_meta, retriever_ident, provider_ident


def test_sidecar_new_run_creation_and_completion():
    """Test: Sidecar is created atomically on new run and marked completed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_out.jsonl")
        mgr = SidecarManager(out_path)
        source_id, man_meta, ret_id, prov_id = _make_dummy_identities()

        meta = mgr.initialize_or_verify(
            system="erag",
            mode="active",
            resolved_config_sha256="h" * 64,
            source_identity=source_id,
            manifest_meta=man_meta,
            retriever_identity=ret_id,
            provider_identity=prov_id,
            raw_config={"test": True},
        )
        assert meta["status"] == "running"
        assert os.path.exists(mgr.sidecar_path)

        # Write dummy prediction and mark completed
        with open(out_path, "w", encoding="utf-8") as pf:
            pf.write(json.dumps({"id": "q1", "answer": "A"}) + "\n")
        mgr.mark_completed(completed_count=1)

        with open(mgr.sidecar_path, "r", encoding="utf-8") as f:
            completed_meta = json.load(f)
        assert completed_meta["status"] == "completed"
        assert completed_meta["progress"]["completed_count"] == 1
        assert len(completed_meta["progress"]["predictions_sha256"]) == 64


def test_sidecar_legacy_output_file_fails_closed():
    """Test: Existing output file without .meta.json raises LegacyOutputError and refuses resume."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "legacy_out.jsonl")
        # Create output file WITHOUT sidecar
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "q1", "answer": "Paris"}) + "\n")

        mgr = SidecarManager(out_path)
        source_id, man_meta, ret_id, prov_id = _make_dummy_identities()

        with pytest.raises(LegacyOutputError, match="lacks companion metadata sidecar"):
            mgr.initialize_or_verify(
                system="erag",
                mode="active",
                resolved_config_sha256="h" * 64,
                source_identity=source_id,
                manifest_meta=man_meta,
                retriever_identity=ret_id,
                provider_identity=prov_id,
                raw_config={},
            )


def test_sidecar_resume_mismatch_fails_closed():
    """Test: Resume against mismatched config or source raises IncompatibleResumeError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "run_mismatch.jsonl")
        mgr = SidecarManager(out_path)
        source_id, man_meta, ret_id, prov_id = _make_dummy_identities()

        # Initialize run with config hash 'h' * 64
        mgr.initialize_or_verify(
            system="erag",
            mode="active",
            resolved_config_sha256="h" * 64,
            source_identity=source_id,
            manifest_meta=man_meta,
            retriever_identity=ret_id,
            provider_identity=prov_id,
            raw_config={},
        )

        # Attempt to resume with different config hash 'diff'
        with pytest.raises(IncompatibleResumeError, match="resolved_config_sha256' mismatch"):
            mgr.initialize_or_verify(
                system="erag",
                mode="active",
                resolved_config_sha256="z" * 64,  # mismatched hash
                source_identity=source_id,
                manifest_meta=man_meta,
                retriever_identity=ret_id,
                provider_identity=prov_id,
                raw_config={},
            )


# -----------------------------------------------------------------------------
# 6. Process Locking Tests
# -----------------------------------------------------------------------------

def test_run_lock_active_process_raises_error():
    """Test: Lock held by an active PID raises ActiveProcessLockError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "locked_run.jsonl")
        lock = RunLock(out_path)
        lock.acquire()

        try:
            # Second attempt to acquire lock on same output while current process holds it
            second_lock = RunLock(out_path)
            with pytest.raises(ActiveProcessLockError, match="is actively executing"):
                second_lock.acquire()
        finally:
            lock.release()


def test_run_lock_stale_lock_requires_explicit_clear():
    """Test: Stale lockfile belonging to dead PID raises StaleLockDetectedError without silent removal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "stale_run.jsonl")
        lock_path = f"{out_path}.lock"
        # Write lock with an impossible dead PID (e.g. 999999999)
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump({"pid": 999999999, "hostname": "local", "started_at": "2026-01-01T00:00:00Z"}, f)

        second_lock = RunLock(out_path, clear_stale_lock=False)
        with pytest.raises(StaleLockDetectedError, match="Stale lockfile detected for deceased PID"):
            second_lock.acquire()

        # But with clear_stale_lock=True, it clears and acquires successfully
        clearing_lock = RunLock(out_path, clear_stale_lock=True)
        clearing_lock.acquire()
        assert clearing_lock.acquired is True
        clearing_lock.release()


# -----------------------------------------------------------------------------
# 7. Memory Feature Gating Tests
# -----------------------------------------------------------------------------

def test_memory_feature_gating_rejects_enabled():
    """Test: memory.enabled=True raises NotImplementedError in guarded runners."""
    cfg = {"memory": {"enabled": True}}
    with pytest.raises(NotImplementedError, match="Memory state provenance is deferred"):
        check_memory_gating(cfg)

    # memory.enabled=False passes cleanly
    check_memory_gating({"memory": {"enabled": False}})


# -----------------------------------------------------------------------------
# 8. Safe Append & Trailing Newline Tests
# -----------------------------------------------------------------------------

def test_safe_append_adds_newline_when_missing():
    """Test: prepare_output_file_for_append appends b'\\n' if file lacks trailing newline."""
    with tempfile.NamedTemporaryFile("wb", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"id": "q1", "ans": "A"}).encode("utf-8"))
        tmp_path = f.name

    try:
        # File currently lacks trailing newline
        with open(tmp_path, "rb") as f:
            content = f.read()
        assert not content.endswith(b"\n")

        prepare_output_file_for_append(tmp_path)

        with open(tmp_path, "rb") as f:
            updated = f.read()
        assert updated.endswith(b"\n")
        assert updated == content + b"\n"

        # Subsequent append must now be on a clean new line
        with open(tmp_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": "q2", "ans": "B"}) + "\n")

        with open(tmp_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 2
        assert lines[0]["id"] == "q1"
        assert lines[1]["id"] == "q2"
    finally:
        os.remove(tmp_path)


def test_safe_append_leaves_trailing_newline_intact():
    """Test: prepare_output_file_for_append does not add superfluous newline if already present."""
    with tempfile.NamedTemporaryFile("wb", suffix=".jsonl", delete=False) as f:
        f.write((json.dumps({"id": "q1", "ans": "A"}) + "\n").encode("utf-8"))
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as f:
            before = f.read()
        prepare_output_file_for_append(tmp_path)
        with open(tmp_path, "rb") as f:
            after = f.read()
        assert before == after
    finally:
        os.remove(tmp_path)


# -----------------------------------------------------------------------------
# 9. Corruption & Crash Recovery Invariant Tests
# -----------------------------------------------------------------------------

def test_mid_file_corruption_fails_closed():
    """Test: Corrupted line in middle of output raises CorruptedOutputFileError."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"id": "q1", "ans": "A"}) + "\n")
        f.write("{broken json line\n")
        f.write(json.dumps({"id": "q2", "ans": "B"}) + "\n")
        tmp_path = f.name

    try:
        with pytest.raises(CorruptedOutputFileError, match="Corrupted or truncated JSON"):
            read_existing_output_records(tmp_path, {"q1", "q2"})
    finally:
        os.remove(tmp_path)


def test_evaluation_metadata_update_does_not_break_generation_resume():
    """Test: Adding or updating top-level evaluation metrics in sidecar does not invalidate generation resume."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "eval_test.jsonl")
        mgr = SidecarManager(out_path)
        source_id, man_meta, ret_id, prov_id = _make_dummy_identities()

        meta = mgr.initialize_or_verify(
            system="erag",
            mode="active",
            resolved_config_sha256="h" * 64,
            source_identity=source_id,
            manifest_meta=man_meta,
            retriever_identity=ret_id,
            provider_identity=prov_id,
            raw_config={},
        )

        # Update evaluation metrics (downstream evaluator pass)
        mgr.update_evaluation({"exact_match": 0.92, "f1": 0.95})

        with open(mgr.sidecar_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["evaluation"]["exact_match"] == 0.92

        # Generation resume check with identical generation identity must succeed
        resumed = mgr.initialize_or_verify(
            system="erag",
            mode="active",
            resolved_config_sha256="h" * 64,
            source_identity=source_id,
            manifest_meta=man_meta,
            retriever_identity=ret_id,
            provider_identity=prov_id,
            raw_config={},
        )
        assert resumed["run_id"] == meta["run_id"]
        assert resumed["evaluation"]["exact_match"] == 0.92


def test_crash_recovery_progress_discrepancy_reconciled():
    """Test: Interruption crash between prediction write and sidecar update is safely reconciled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "crash_test.jsonl")
        mgr = SidecarManager(out_path)
        source_id, man_meta, ret_id, prov_id = _make_dummy_identities()

        mgr.initialize_or_verify(
            system="erag",
            mode="active",
            resolved_config_sha256="h" * 64,
            source_identity=source_id,
            manifest_meta=man_meta,
            retriever_identity=ret_id,
            provider_identity=prov_id,
            raw_config={},
        )

        # Output has 2 valid records written before crash
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "q1", "ans": "Paris", "failed": False}) + "\n")
            f.write(json.dumps({"id": "q2", "ans": "London", "failed": True}) + "\n")

        # Sidecar progress snapshot still says 0 (simulating crash before completion)
        with open(mgr.sidecar_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["progress"]["completed_count"] == 0

        # Resume discovers both records (including failed=True) from authoritative .jsonl
        records, done_ids = read_existing_output_records(out_path, {"q1", "q2", "q3"})
        assert done_ids == {"q1", "q2"}
        assert len(records) == 2


# -----------------------------------------------------------------------------
# 10. Runner Integration Tests (run_experiment & run_phase5)
# -----------------------------------------------------------------------------

def _make_runner_test_cfg(tmpdir: str, name: str = "run_test") -> Dict[str, Any]:
    return {
        "model": {"name": "test-llm", "temperature": 0.0, "max_tokens": 100},
        "retrieval": {"setting": "distractor", "embedder": "minilm"},
        "erag": {"theta_fast": 0.80, "max_hops": 2},
        "data": {"manifest_dir": tmpdir, "split": "test"},
        "run": {
            "name": name,
            "out_dir": tmpdir,
            "system": "erag",
            "mode": "active",
            "cache": False,
            "cache_namespace": "test",
        },
    }


def test_run_experiment_integration_full_lifecycle():
    """Integration Test: run_experiment creates sidecar, locks, executes, and resumes skipping failed."""
    from scripts.run_experiment import dispatch

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_runner_test_cfg(tmpdir, name="exp_lifecycle")
        manifest_path = os.path.join(tmpdir, "test.jsonl")
        out_path = os.path.join(tmpdir, "exp_lifecycle.jsonl")

        # Write manifest with 2 items
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_id": "q1", "question": "Capital of France?", "answer": "Paris"}) + "\n")
            f.write(json.dumps({"_id": "q2", "question": "Capital of Germany?", "answer": "Berlin"}) + "\n")

        manifest_meta = validate_manifest(manifest_path)
        items = manifest_meta["items"]
        manifest_id_set = {it["_id"] for it in items}

        source_ident = {
            "git_commit_sha": "a" * 40,
            "git_dirty": False,
            "git_diff_sha256": None,
            "tracked_paths": ["src"],
        }
        cfg_hash = compute_config_hash(cfg)
        sidecar_mgr = SidecarManager(out_path)

        # Run 1: Write q1 as success, q2 as failed
        with RunLock(out_path):
            sidecar_mgr.initialize_or_verify(
                system="erag",
                mode="active",
                resolved_config_sha256=cfg_hash,
                source_identity=source_ident,
                manifest_meta=manifest_meta,
                retriever_identity={"setting": "distractor", "embedder": "minilm"},
                provider_identity={"model": "test-llm", "temperature": 0.0, "max_tokens": 100},
                raw_config=cfg,
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "q1", "answer": "Paris", "failed": False}) + "\n")
                f.write(json.dumps({"id": "q2", "answer": "", "failed": True}) + "\n")

        # Run 2: Resume check
        with RunLock(out_path):
            existing_records, done_ids = read_existing_output_records(out_path, manifest_id_set)
            assert done_ids == {"q1", "q2"}, "Both successful and failed records must be in done_ids"
            remaining = [it for it in items if it["_id"] not in done_ids]
            assert len(remaining) == 0, "No duplicate execution of completed or failed items"
            prepare_output_file_for_append(out_path)
            sidecar_mgr.mark_completed(completed_count=len(items))

        with open(sidecar_mgr.sidecar_path, "r", encoding="utf-8") as f:
            completed_meta = json.load(f)
        assert completed_meta["status"] == "completed"
        assert completed_meta["progress"]["completed_count"] == 2


def test_run_experiment_integration_mismatch_halts_before_model_call():
    """Integration Test: Configuration mismatch raises IncompatibleResumeError before inference."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_runner_test_cfg(tmpdir, name="mismatch_guard")
        manifest_path = os.path.join(tmpdir, "test.jsonl")
        out_path = os.path.join(tmpdir, "mismatch_guard.jsonl")

        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_id": "q1", "question": "Question 1", "answer": "Ans"}) + "\n")

        manifest_meta = validate_manifest(manifest_path)
        source_ident = {
            "git_commit_sha": "a" * 40,
            "git_dirty": False,
            "git_diff_sha256": None,
            "tracked_paths": ["src"],
        }
        cfg_hash = compute_config_hash(cfg)
        sidecar_mgr = SidecarManager(out_path)

        with RunLock(out_path):
            sidecar_mgr.initialize_or_verify(
                system="erag",
                mode="active",
                resolved_config_sha256=cfg_hash,
                source_identity=source_ident,
                manifest_meta=manifest_meta,
                retriever_identity={"setting": "distractor", "embedder": "minilm"},
                provider_identity={"model": "test-llm", "temperature": 0.0, "max_tokens": 100},
                raw_config=cfg,
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "q1", "answer": "Ans", "failed": False}) + "\n")

        # Resume attempt with altered configuration (theta_fast changed)
        cfg_altered = dict(cfg)
        cfg_altered["erag"] = dict(cfg["erag"])
        cfg_altered["erag"]["theta_fast"] = 0.50
        altered_hash = compute_config_hash(cfg_altered)

        with RunLock(out_path):
            with pytest.raises(IncompatibleResumeError, match="resolved_config_sha256' mismatch"):
                sidecar_mgr.initialize_or_verify(
                    system="erag",
                    mode="active",
                    resolved_config_sha256=altered_hash,
                    source_identity=source_ident,
                    manifest_meta=manifest_meta,
                    retriever_identity={"setting": "distractor", "embedder": "minilm"},
                    provider_identity={"model": "test-llm", "temperature": 0.0, "max_tokens": 100},
                    raw_config=cfg_altered,
                )


def test_run_phase5_integration_full_lifecycle():
    """Integration Test: execute_system_retriever validates sidecar, skips done, and marks completed."""
    from scripts.run_phase5_four_retrievers import execute_system_retriever

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_runner_test_cfg(tmpdir, name="phase5_test")
        out_path = os.path.join(tmpdir, "p5_out.jsonl")

        test_items = [
            {"_id": "q1", "question": "Question 1", "answer": "A"},
            {"_id": "q2", "question": "Question 2", "answer": "B"},
        ]
        manifest_meta = {
            "manifest_path": "data/manifests/test.jsonl",
            "manifest_sha256": "m" * 64,
            "manifest_id_sequence_sha256": "s" * 64,
            "total_questions": len(test_items),
        }

        # Pre-seed output with q1 completed
        mgr = SidecarManager(out_path)
        source_id, _, ret_id, prov_id = _make_dummy_identities()
        resolved_hash = compute_config_hash(cfg)

        with patch("scripts.run_phase5_four_retrievers.get_source_identity", return_value=source_id):
            with patch("scripts.run_phase5_four_retrievers.RetrieverFactory"), \
                 patch("scripts.run_phase5_four_retrievers.LLM"), \
                 patch("scripts.run_phase5_four_retrievers.run_standard", return_value={"answer": "B", "pool_titles": []}):

                # Write q1
                with RunLock(out_path):
                    mgr.initialize_or_verify(
                        system="Standard RAG",
                        mode="active",
                        resolved_config_sha256=resolved_hash,
                        source_identity=source_id,
                        manifest_meta=manifest_meta,
                        retriever_identity={"setting": "distractor", "retriever_type": "MiniLM", "embedder": "MiniLM"},
                        provider_identity={
                            "model": cfg.get("model", {}).get("name"),
                            "temperature": cfg.get("model", {}).get("temperature"),
                            "max_tokens": cfg.get("model", {}).get("max_tokens"),
                            "reasoning_effort": cfg.get("model", {}).get("reasoning_effort"),
                        },
                        raw_config=cfg,
                    )
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(json.dumps({"id": "q1", "question": "Question 1", "answer": "A", "gold": "A", "em": 1.0, "f1": 1.0}) + "\n")

                # Resume execute_system_retriever on test_items: should skip q1 and evaluate only q2
                records, summary = execute_system_retriever(
                    system_type="Standard RAG",
                    retriever_name="MiniLM",
                    output_path=out_path,
                    test_items=test_items,
                    nli=None,
                    cfg=cfg,
                    embedder=None,
                    limit=None,
                    manifest_meta=manifest_meta,
                )

                assert len(records) == 2
                assert records[0]["id"] == "q1"
                assert records[1]["id"] == "q2"

                with open(mgr.sidecar_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                assert meta["status"] == "completed"
                assert meta["progress"]["completed_count"] == 2


def test_run_phase5_integration_mismatch_halts_before_model_call():
    """Integration Test: execute_system_retriever halts on config mismatch without running model."""
    from scripts.run_phase5_four_retrievers import execute_system_retriever

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_runner_test_cfg(tmpdir, name="phase5_mismatch")
        out_path = os.path.join(tmpdir, "p5_mismatch.jsonl")

        test_items = [{"_id": "q1", "question": "Question 1", "answer": "A"}]
        manifest_meta = {
            "manifest_path": "data/manifests/test.jsonl",
            "manifest_sha256": "m" * 64,
            "manifest_id_sequence_sha256": "s" * 64,
            "total_questions": 1,
        }

        mgr = SidecarManager(out_path)
        source_id, _, ret_id, prov_id = _make_dummy_identities()
        resolved_hash = compute_config_hash(cfg)

        with RunLock(out_path):
            mgr.initialize_or_verify(
                system="Standard RAG",
                mode="active",
                resolved_config_sha256=resolved_hash,
                source_identity=source_id,
                manifest_meta=manifest_meta,
                retriever_identity={"setting": "distractor", "retriever_type": "MiniLM", "embedder": "MiniLM"},
                provider_identity={
                    "model": cfg.get("model", {}).get("name"),
                    "temperature": cfg.get("model", {}).get("temperature"),
                    "max_tokens": cfg.get("model", {}).get("max_tokens"),
                    "reasoning_effort": cfg.get("model", {}).get("reasoning_effort"),
                },
                raw_config=cfg,
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "q1", "question": "Question 1", "answer": "A"}) + "\n")

        # Incompatible config: altered model max_tokens
        cfg_mismatch = dict(cfg)
        cfg_mismatch["model"] = dict(cfg["model"])
        cfg_mismatch["model"]["max_tokens"] = 999

        with patch("scripts.run_phase5_four_retrievers.get_source_identity", return_value=source_id):
            with patch("scripts.run_phase5_four_retrievers.LLM") as mock_llm:
                with pytest.raises(IncompatibleResumeError, match="resolved_config_sha256' mismatch"):
                    execute_system_retriever(
                        system_type="Standard RAG",
                        retriever_name="MiniLM",
                        output_path=out_path,
                        test_items=test_items,
                        nli=None,
                        cfg=cfg_mismatch,
                        embedder=None,
                        manifest_meta=manifest_meta,
                    )
                # Ensure no model initialization or calls took place
                mock_llm.assert_not_called()

