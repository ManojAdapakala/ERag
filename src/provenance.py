"""Experiment provenance, manifest validation, process locking, and fail-closed resume safeguards."""

import errno
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class IncompatibleResumeError(RuntimeError):
    """Raised when an existing run cannot be resumed due to configuration or identity mismatch."""
    pass


class CorruptedOutputFileError(RuntimeError):
    """Raised when an existing output file has malformed JSON or duplicate question IDs."""
    pass


class LegacyOutputError(RuntimeError):
    """Raised when an existing output file lacks a companion metadata sidecar."""
    pass


class UntrackedCodeError(RuntimeError):
    """Raised when untracked code exists in execution-relevant directories."""
    pass


class ActiveProcessLockError(RuntimeError):
    """Raised when another active process is currently holding the run lock."""
    pass


class StaleLockDetectedError(RuntimeError):
    """Raised when an existing lockfile belongs to a deceased PID."""
    pass


# -----------------------------------------------------------------------------
# 1. Deterministic Configuration Hashing
# -----------------------------------------------------------------------------

def canonicalize_value(val: Any) -> Any:
    """Recursively canonicalize nested dictionaries, lists, and primitives for stable hashing."""
    if isinstance(val, dict):
        return {k: canonicalize_value(v) for k, v in sorted(val.items())}
    if isinstance(val, (list, tuple)):
        return [canonicalize_value(v) for v in val]
    return val


def compute_config_hash(cfg: Dict[str, Any]) -> str:
    """Compute full 64-character SHA-256 digest of output-relevant configuration.
    
    Excludes cosmetic run-level keys (such as display names, out_dir, logging).
    Captures all parameters governing retrieval, model generation, epistemic graph,
    confidence estimation, and abstention policies.
    """
    relevant = {}
    for section in ("retrieval", "model", "erag", "ircot", "nli", "memory", "prompts"):
        if section in cfg:
            relevant[section] = cfg[section]
    
    # Include output-relevant execution flags under 'run'
    if "run" in cfg:
        run_sec = cfg["run"]
        relevant["run"] = {
            k: run_sec[k] for k in ("cache", "cache_namespace", "cache_path")
            if k in run_sec
        }

    canonical = canonicalize_value(relevant)
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# 2. Source Identity & Untracked Code Verification
# -----------------------------------------------------------------------------

def get_source_identity(
    repo_root: Optional[str] = None,
    relevant_dirs: Tuple[str, ...] = ("src", "scripts", "configs"),
) -> Dict[str, Any]:
    """Capture exact source provenance, rejecting untracked code in execution-relevant paths.
    
    Returns:
        dict containing:
            - git_commit_sha: Full 40-character SHA of HEAD.
            - git_dirty: bool indicating if tracked files in relevant_dirs have modifications.
            - git_diff_sha256: Full 64-character SHA-256 of git diff HEAD across relevant_dirs (or None if clean).
    """
    root = repo_root or os.getcwd()
    
    # 1. Check for untracked code files (.py, .yaml, .yml) in relevant directories
    cmd_status = ["git", "status", "--porcelain", "--"] + list(relevant_dirs)
    try:
        proc_status = subprocess.run(
            cmd_status, cwd=root, capture_output=True, text=True, check=True
        )
    except Exception as ex:
        raise RuntimeError(f"Failed to inspect git status: {ex}") from ex

    untracked_files = []
    for line in proc_status.stdout.splitlines():
        if line.startswith("??"):
            path = line[3:].strip()
            if path.endswith((".py", ".yaml", ".yml")):
                untracked_files.append(path)

    if untracked_files:
        raise UntrackedCodeError(
            f"Untracked execution-relevant code found in {relevant_dirs}: {untracked_files}. "
            "Track, stash, or remove these files to ensure exact source reproducibility."
        )

    # 2. Obtain full 40-character commit SHA of HEAD
    try:
        proc_rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
        commit_sha = proc_rev.stdout.strip()
    except Exception as ex:
        raise RuntimeError(f"Failed to get git commit SHA: {ex}") from ex

    # 3. Check for modified tracked files in relevant directories
    cmd_diff = ["git", "diff", "HEAD", "--"] + list(relevant_dirs)
    try:
        proc_diff = subprocess.run(
            cmd_diff, cwd=root, capture_output=True, check=True
        )
        diff_bytes = proc_diff.stdout
    except Exception as ex:
        raise RuntimeError(f"Failed to compute git diff: {ex}") from ex

    if diff_bytes.strip():
        git_dirty = True
        git_diff_sha256 = hashlib.sha256(diff_bytes).hexdigest()
    else:
        git_dirty = False
        git_diff_sha256 = None

    return {
        "git_commit_sha": commit_sha,
        "git_dirty": git_dirty,
        "git_diff_sha256": git_diff_sha256,
        "tracked_paths": list(relevant_dirs),
    }


# -----------------------------------------------------------------------------
# 3. Manifest Pre-Flight Validation
# -----------------------------------------------------------------------------

def validate_manifest(manifest_path: str) -> Dict[str, Any]:
    """Validate input manifest for non-empty unique question IDs and calculate SHA-256 digests.
    
    Returns:
        dict containing:
            - items: List of parsed manifest item dictionaries.
            - manifest_sha256: Full 64-character SHA-256 of raw file content.
            - manifest_id_sequence_sha256: Full 64-character SHA-256 of ordered question ID tuple.
            - total_questions: Number of questions in manifest.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, "rb") as f:
        raw_bytes = f.read()

    manifest_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    lines = raw_bytes.decode("utf-8").splitlines()

    items = []
    seen_ids = set()
    ordered_ids = []

    for idx, line in enumerate(lines, start=1):
        line_str = line.strip()
        if not line_str:
            continue
        try:
            item = json.loads(line_str)
        except Exception as ex:
            raise ValueError(f"Malformed JSON in manifest '{manifest_path}' at line {idx}: {ex}") from ex

        qid = item.get("_id")
        if not qid or not str(qid).strip():
            raise ValueError(f"Manifest record in '{manifest_path}' at line {idx} missing non-empty '_id'")

        if qid in seen_ids:
            raise ValueError(f"Duplicate question ID '{qid}' detected in manifest '{manifest_path}' at line {idx}")

        seen_ids.add(qid)
        ordered_ids.append(qid)
        items.append(item)

    id_seq_serialized = json.dumps(ordered_ids, separators=(",", ":"))
    id_seq_hash = hashlib.sha256(id_seq_serialized.encode("utf-8")).hexdigest()

    return {
        "items": items,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "manifest_id_sequence_sha256": id_seq_hash,
        "total_questions": len(items),
    }


# -----------------------------------------------------------------------------
# 4. Fail-Closed Output Reading & Invariant Validation
# -----------------------------------------------------------------------------

def read_existing_output_records(
    out_path: str, manifest_id_set: Set[str]
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Read existing output JSONL, validating record format, ID uniqueness, and manifest membership.
    
    Treats every valid record, including records where failed=True, as completed.
    Fails closed on any corrupted, truncated, duplicate, or unparseable line.
    Preserves valid trailing records even if they lack a trailing newline character.
    
    Returns:
        (valid_records, done_ids)
    """
    if not os.path.exists(out_path):
        return [], set()

    with open(out_path, "rb") as f:
        raw_content = f.read()

    if not raw_content.strip():
        return [], set()

    # Split lines preserving exact line entries (splitlines handles missing trailing newline cleanly)
    text_content = raw_content.decode("utf-8")
    lines = text_content.splitlines()

    records = []
    done_ids = set()

    for line_idx, line in enumerate(lines, start=1):
        line_str = line.strip()
        if not line_str:
            continue
        try:
            record = json.loads(line_str)
        except Exception as ex:
            raise CorruptedOutputFileError(
                f"Corrupted or truncated JSON at line {line_idx} in output file '{out_path}': {ex}. "
                "Resume fails closed to prevent output corruption. Manual inspection required."
            ) from ex

        qid = record.get("id")
        if not qid:
            raise CorruptedOutputFileError(
                f"Missing question 'id' field at line {line_idx} in output file '{out_path}'"
            )

        if qid not in manifest_id_set:
            raise IncompatibleResumeError(
                f"Output file '{out_path}' contains question ID '{qid}' at line {line_idx} "
                "which is not present in the active manifest. Resume rejected."
            )

        if qid in done_ids:
            raise CorruptedOutputFileError(
                f"Duplicate question ID '{qid}' detected at line {line_idx} in output file '{out_path}'. "
                "Resume rejected to prevent multiple records per question."
            )

        done_ids.add(qid)
        records.append(record)

    return records, done_ids


def prepare_output_file_for_append(out_path: str) -> None:
    """Ensure output file ends with a newline before subsequent appends.
    
    If the file exists and is non-empty, inspects the final byte.
    If the final byte is not b'\\n', appends exactly one b'\\n'.
    Never truncates, re-encodes, or rewrites existing records.
    """
    if not os.path.exists(out_path):
        return
    file_size = os.path.getsize(out_path)
    if file_size == 0:
        return
    with open(out_path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        last_byte = f.read(1)
        if last_byte != b"\n":
            f.seek(0, os.SEEK_END)
            f.write(b"\n")


# -----------------------------------------------------------------------------
# 5. Process Locking with Stale-Lock Detection
# -----------------------------------------------------------------------------

def is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is actively running on the local host."""
    if pid <= 0:
        return False
    import sys
    if sys.platform == "win32":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if h:
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError as err:
            return err.errno == errno.EPERM
        except Exception:
            return False


class RunLock:
    """Exclusive filesystem process lock guarding benchmark output files."""

    def __init__(self, out_path: str, clear_stale_lock: bool = False):
        self.out_path = out_path
        self.lock_path = f"{out_path}.lock"
        self.clear_stale_lock = clear_stale_lock
        self.acquired = False

    def acquire(self) -> None:
        if os.path.exists(self.lock_path):
            try:
                with open(self.lock_path, "r", encoding="utf-8") as lf:
                    info = json.load(lf)
                lock_pid = int(info.get("pid", -1))
                hostname = info.get("hostname", "unknown")
                started_at = info.get("started_at", "unknown")
            except Exception:
                lock_pid = -1
                hostname = "unknown"
                started_at = "unknown"

            if lock_pid > 0 and is_pid_alive(lock_pid):
                raise ActiveProcessLockError(
                    f"Process {lock_pid} on {hostname} started at {started_at} is actively executing "
                    f"targeting '{self.out_path}'. Concurrent execution prohibited."
                )

            # Process is deceased: stale lock detected
            if not self.clear_stale_lock:
                raise StaleLockDetectedError(
                    f"Stale lockfile detected for deceased PID {lock_pid} started at {started_at} "
                    f"targeting '{self.out_path}'. Refusing to remove lock silently. "
                    "Inspect the crashed run or pass clear_stale_lock=True to clear."
                )
            # User explicitly authorized clearing stale lock
            try:
                os.remove(self.lock_path)
            except OSError:
                pass

        # Write lock exclusively
        payload = {
            "pid": os.getpid(),
            "hostname": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "localhost",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(self.lock_path, flags)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            self.acquired = True
        except FileExistsError:
            raise ActiveProcessLockError(
                f"Failed to acquire lock '{self.lock_path}' due to race condition."
            )

    def release(self) -> None:
        if self.acquired:
            try:
                if os.path.exists(self.lock_path):
                    os.remove(self.lock_path)
            except OSError:
                pass
            self.acquired = False

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


# -----------------------------------------------------------------------------
# 6. Sidecar Metadata Management
# -----------------------------------------------------------------------------

class SidecarManager:
    """Manages immutable run identity and execution metadata sidecars (<out_path>.meta.json)."""

    SCHEMA_VERSION = "2.0"

    def __init__(self, out_path: str):
        self.out_path = out_path
        self.sidecar_path = f"{out_path}.meta.json"

    def initialize_or_verify(
        self,
        system: str,
        mode: str,
        resolved_config_sha256: str,
        source_identity: Dict[str, Any],
        manifest_meta: Dict[str, Any],
        retriever_identity: Dict[str, Any],
        provider_identity: Dict[str, Any],
        raw_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify immutable identity on resume or initialize sidecar for a new run.
        
        Fails closed on missing sidecars (legacy files), config mismatches, source mismatches,
        or manifest discrepancies.
        """
        runtime_identity = {
            "system": system,
            "mode": mode,
            "resolved_config_sha256": resolved_config_sha256,
            "source_identity": source_identity,
            "manifest_identity": {
                "manifest_path": manifest_meta["manifest_path"],
                "manifest_sha256": manifest_meta["manifest_sha256"],
                "manifest_id_sequence_sha256": manifest_meta["manifest_id_sequence_sha256"],
                "total_questions": manifest_meta["total_questions"],
            },
            "retriever_identity": retriever_identity,
            "provider_identity": provider_identity,
        }

        out_exists = os.path.exists(self.out_path)
        sidecar_exists = os.path.exists(self.sidecar_path)

        if out_exists and not sidecar_exists:
            raise LegacyOutputError(
                f"Existing output file '{self.out_path}' lacks companion metadata sidecar '{self.sidecar_path}'. "
                "Resume fails closed to protect historical experiment records. Specify a new output path."
            )

        if sidecar_exists:
            # Resume verification
            with open(self.sidecar_path, "r", encoding="utf-8") as f:
                saved = json.load(f)

            if saved.get("schema_version") != self.SCHEMA_VERSION:
                raise IncompatibleResumeError(
                    f"Unsupported sidecar schema version '{saved.get('schema_version')}'; expected '{self.SCHEMA_VERSION}'"
                )

            saved_ident = saved.get("immutable_identity", {})
            self._verify_identity_match(saved_ident, runtime_identity)
            return saved

        # New run initialization
        run_id = f"{system}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        meta = {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "immutable_identity": runtime_identity,
            "raw_config": raw_config,
            "progress": {
                "completed_count": 0,
                "last_updated_at": datetime.now(timezone.utc).isoformat(),
            },
            "evaluation": {},
        }
        self._atomic_write(meta)
        return meta

    def _verify_identity_match(self, saved: Dict[str, Any], runtime: Dict[str, Any]) -> None:
        """Strictly compare all immutable identity fields, raising on any mismatch."""
        checks = [
            ("system", saved.get("system"), runtime.get("system")),
            ("mode", saved.get("mode"), runtime.get("mode")),
            ("resolved_config_sha256", saved.get("resolved_config_sha256"), runtime.get("resolved_config_sha256")),
            (
                "git_commit_sha",
                saved.get("source_identity", {}).get("git_commit_sha"),
                runtime.get("source_identity", {}).get("git_commit_sha"),
            ),
            (
                "git_diff_sha256",
                saved.get("source_identity", {}).get("git_diff_sha256"),
                runtime.get("source_identity", {}).get("git_diff_sha256"),
            ),
            (
                "manifest_sha256",
                saved.get("manifest_identity", {}).get("manifest_sha256"),
                runtime.get("manifest_identity", {}).get("manifest_sha256"),
            ),
            (
                "manifest_id_sequence_sha256",
                saved.get("manifest_identity", {}).get("manifest_id_sequence_sha256"),
                runtime.get("manifest_identity", {}).get("manifest_id_sequence_sha256"),
            ),
            (
                "retriever_identity",
                saved.get("retriever_identity"),
                runtime.get("retriever_identity"),
            ),
            (
                "provider_identity",
                saved.get("provider_identity"),
                runtime.get("provider_identity"),
            ),
        ]

        for field_name, s_val, r_val in checks:
            if s_val != r_val:
                raise IncompatibleResumeError(
                    f"Cannot resume run: immutable field '{field_name}' mismatch.\n"
                    f"  Existing sidecar: {s_val}\n"
                    f"  Runtime value:    {r_val}\n"
                    "Resuming against mismatched configuration or source is prohibited. Provide a new output path."
                )

    def mark_completed(self, completed_count: int) -> None:
        """Update sidecar upon successful completion of all questions."""
        if not os.path.exists(self.sidecar_path):
            return

        with open(self.sidecar_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Compute predictions SHA-256
        pred_hash = None
        if os.path.exists(self.out_path):
            with open(self.out_path, "rb") as pf:
                pred_hash = hashlib.sha256(pf.read()).hexdigest()

        data["status"] = "completed"
        data["progress"] = {
            "completed_count": completed_count,
            "last_updated_at": datetime.now(timezone.utc).isoformat(),
            "predictions_sha256": pred_hash,
        }
        self._atomic_write(data)

    def update_evaluation(self, eval_data: Dict[str, Any]) -> None:
        """Update top-level evaluation section without affecting immutable generation identity."""
        if not os.path.exists(self.sidecar_path):
            return

        with open(self.sidecar_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data.setdefault("evaluation", {}).update(eval_data)
        self._atomic_write(data)

    def _atomic_write(self, data: Dict[str, Any]) -> None:
        """Atomically replace sidecar file using a temporary file."""
        tmp_path = f"{self.sidecar_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.sidecar_path)


# -----------------------------------------------------------------------------
# 7. Memory Feature Gating
# -----------------------------------------------------------------------------

def check_memory_gating(cfg: Dict[str, Any]) -> None:
    """Explicitly reject memory-enabled runs in guarded runners."""
    if cfg.get("memory", {}).get("enabled"):
        raise NotImplementedError(
            "Memory state provenance is deferred and not permitted in guarded runners. "
            "Set memory.enabled=False to execute."
        )
