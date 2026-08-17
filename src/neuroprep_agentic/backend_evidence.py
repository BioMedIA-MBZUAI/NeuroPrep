from __future__ import annotations

import glob
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import stat
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .settings import validated_cluster_path


class EvidenceValidationError(ValueError):
    """Raised when backend evidence cannot be trusted for a target plan."""


def _validated_sha256(value: str, label: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase hexadecimal SHA-256 digest")
    return value


def _validated_git_commit(value: str, label: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{label} must be a full lowercase hexadecimal git commit")
    return value


class FileIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    size: int = Field(ge=1)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def valid_digest(cls, value: str) -> str:
        return _validated_sha256(value, "sha256")


class ProtocolSignature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    acquisition_count: int = Field(ge=0)
    modality_plugins: list[str]
    source_formats: list[str]
    signature_sha256: str

    @field_validator("signature_sha256")
    @classmethod
    def valid_signature_digest(cls, value: str) -> str:
        return _validated_sha256(value, "protocol signature")


class PilotGitProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commit: str | None = None
    verification: Literal["embedded", "unknown"]
    sources: list[str] = Field(default_factory=list)
    detail: str

    @model_validator(mode="after")
    def internally_consistent(self) -> PilotGitProvenance:
        if self.verification == "unknown":
            if self.commit is not None or self.sources:
                raise ValueError("unknown pilot git provenance cannot claim a commit")
        else:
            if self.commit is None or not self.sources:
                raise ValueError("embedded pilot git provenance requires sources and a commit")
            _validated_git_commit(self.commit, "pilot git commit")
        return self


class PilotAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    plan: FileIdentity
    state: FileIdentity
    release_summary: FileIdentity
    artifact_index: FileIdentity
    tensor_index: FileIdentity
    release_ready: bool
    task_counts: dict[str, int]
    artifact_count: int = Field(ge=1)
    tensor_count: int = Field(ge=0)
    artifact_audit_sha256: str
    protocol: ProtocolSignature
    git: PilotGitProvenance

    @field_validator("artifact_audit_sha256")
    @classmethod
    def valid_artifact_digest(cls, value: str) -> str:
        return _validated_sha256(value, "pilot artifact audit")

    @model_validator(mode="after")
    def successful_release_and_protocol(self) -> PilotAudit:
        if not self.release_ready:
            raise ValueError("pilot release must be ready")
        if self.protocol.dataset_id != self.dataset_id:
            raise ValueError("pilot protocol dataset differs from pilot dataset")
        return self


class SmokeTestIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_id: str
    capability: str
    package_id: str
    executor_id: str


class TargetBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    qualification_plan: FileIdentity
    protocol: ProtocolSignature
    smoke_test_ids: list[str] = Field(min_length=1)
    smoke_tests: list[SmokeTestIdentity] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_smoke_tests(self) -> TargetBinding:
        ids = [item.test_id for item in self.smoke_tests]
        if len(ids) != len(set(ids)):
            raise ValueError("target smoke test ids must be unique")
        if self.smoke_test_ids != sorted(ids):
            raise ValueError("target smoke ids must equal target smoke test specifications")
        if self.protocol.dataset_id != self.dataset_id:
            raise ValueError("target protocol dataset differs from target dataset")
        return self


class BackendTestEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_id: str
    capability: str
    package_id: str
    executor_id: str
    status: Literal["passed", "failed"]
    executor_signature_sha256: str
    pilot_task_count: int = Field(ge=1)
    artifact_count: int = Field(ge=1)
    artifact_audit_sha256: str
    image: FileIdentity

    @field_validator("executor_signature_sha256", "artifact_audit_sha256")
    @classmethod
    def valid_test_digest(cls, value: str) -> str:
        return _validated_sha256(value, "backend test digest")


class EvidenceSignature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["hmac-sha256"] = "hmac-sha256"
    key_id: str
    payload_sha256: str
    value: str

    @field_validator("key_id")
    @classmethod
    def valid_key_id(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", value):
            raise ValueError("invalid evidence signing key id")
        return value

    @field_validator("payload_sha256", "value")
    @classmethod
    def valid_signature_digest(cls, value: str) -> str:
        return _validated_sha256(value, "evidence signature digest")


class BackendEvidenceDocument(BaseModel):
    """Signed, target-bound backend qualification evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    generated_at: str
    catalog_digest: str
    target_git_commit: str
    pilot: PilotAudit
    target: TargetBinding
    tests: list[BackendTestEvidence] = Field(min_length=1)
    signature: EvidenceSignature

    @field_validator("catalog_digest")
    @classmethod
    def valid_catalog_digest(cls, value: str) -> str:
        return _validated_sha256(value, "catalog digest")

    @field_validator("target_git_commit")
    @classmethod
    def valid_target_commit(cls, value: str) -> str:
        return _validated_git_commit(value, "target git commit")

    @model_validator(mode="after")
    def unique_target_tests(self) -> BackendEvidenceDocument:
        ids = [item.test_id for item in self.tests]
        if len(ids) != len(set(ids)):
            raise ValueError("backend evidence test ids must be unique")
        if not set(ids) <= set(self.target.smoke_test_ids):
            raise ValueError("evidence tests must intersect the signed target smoke tests")
        target_specs = {item.test_id: item for item in self.target.smoke_tests}
        for item in self.tests:
            expected = target_specs[item.test_id]
            if (
                item.capability,
                item.package_id,
                item.executor_id,
            ) != (
                expected.capability,
                expected.package_id,
                expected.executor_id,
            ):
                raise ValueError("evidence test identity differs from target smoke test")
        return self


class BackendEvidenceBinding(BaseModel):
    """Small immutable reference embedded in backend and execution plans."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    trust: Literal["verified", "legacy_untrusted"]
    source_path: str
    document_sha256: str
    catalog_digest: str | None = None
    pilot_git_commit: str | None = None
    target_git_commit: str | None = None
    key_id: str | None = None
    pilot_plan_sha256: str | None = None
    target_protocol_sha256: str | None = None
    required_test_ids: list[str] = Field(default_factory=list)
    matched_test_ids: list[str] = Field(default_factory=list)

    @field_validator(
        "document_sha256",
        "catalog_digest",
        "pilot_plan_sha256",
        "target_protocol_sha256",
    )
    @classmethod
    def valid_binding_digest(cls, value: str | None) -> str | None:
        return _validated_sha256(value, "backend evidence binding digest") if value else value

    @field_validator("pilot_git_commit", "target_git_commit")
    @classmethod
    def valid_binding_commit(cls, value: str | None) -> str | None:
        return _validated_git_commit(value, "backend evidence git commit") if value else value


class LoadedBackendEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statuses: dict[str, Literal["passed", "failed"]]
    binding: BackendEvidenceBinding


def _canonical_bytes(value: Any) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: str | Path) -> FileIdentity:
    candidate = Path(path).resolve(strict=True)
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise EvidenceValidationError(f"evidence file is absent or empty: {candidate}")
    return FileIdentity(
        path=str(candidate),
        size=candidate.stat().st_size,
        sha256=sha256_file(candidate),
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).resolve(strict=True)
    loaded = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise EvidenceValidationError(f"expected a JSON object: {candidate}")
    return loaded


def read_signing_key(path: str | Path) -> bytes:
    candidate = validated_cluster_path(str(path), must_exist=True)
    if candidate.is_symlink():
        raise EvidenceValidationError("evidence signing key cannot be a symlink")
    candidate = candidate.resolve(strict=True)
    mode = stat.S_IMODE(candidate.stat().st_mode)
    if mode != 0o600:
        raise EvidenceValidationError("evidence signing key must have mode 0600")
    key = candidate.read_bytes().strip()
    if len(key) < 32:
        raise EvidenceValidationError("evidence signing key must contain at least 32 bytes")
    return key


def create_signing_key(path: str | Path) -> Path:
    """Create an NFS-resident 256-bit key without exposing its contents."""

    destination = validated_cluster_path(str(path))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite evidence signing key: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(secrets.token_hex(32).encode("ascii") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    destination.chmod(0o600)
    return destination


def current_git_commit(repository: str | Path) -> str:
    repository = Path(repository).resolve(strict=True)
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if status.stdout.strip():
        raise EvidenceValidationError("backend evidence requires a clean git worktree")
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    commit = completed.stdout.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise EvidenceValidationError("could not resolve a full git commit")
    return commit


def _pilot_git_provenance(
    pilot_plan: dict[str, Any],
    state_metadata: dict[str, str],
    release: dict[str, Any],
) -> PilotGitProvenance:
    observed: dict[str, str] = {}
    for label, payload in (
        ("pilot_plan", pilot_plan),
        ("pilot_state", state_metadata),
        ("pilot_release", release),
    ):
        value = payload.get("git_commit")
        if value is not None:
            observed[label] = str(value).lower()
    if not observed:
        return PilotGitProvenance(
            verification="unknown",
            sources=[],
            detail=(
                "pilot plan, state metadata, and release summary contain no immutable "
                "git commit; no pilot commit is claimed"
            ),
        )
    if any(not re.fullmatch(r"[0-9a-f]{40}", value) for value in observed.values()):
        raise EvidenceValidationError("pilot provenance contains an invalid git commit")
    if len(set(observed.values())) != 1:
        raise EvidenceValidationError("pilot provenance git commits disagree")
    return PilotGitProvenance(
        commit=next(iter(observed.values())),
        verification="embedded",
        sources=sorted(observed),
        detail="git commit is embedded in hashed immutable pilot artifacts",
    )


def _normalize_value(value: str, plan: dict[str, Any], task: dict[str, Any]) -> str:
    replacements: list[tuple[str, str]] = []
    for key in ("work_dir", "output_root"):
        if task.get(key):
            replacements.append((str(task[key]), "{" + key + "}"))
    if task.get("task_id"):
        replacements.append((str(task["task_id"]), "{task_id}"))
    token_replacements: list[tuple[str, str]] = []
    for key, prefix in (("subject_id", "sub-"), ("session_id", "ses-")):
        raw = task.get(key)
        if not raw:
            continue
        raw = str(raw)
        replacements.append((raw, "{" + key + "}"))
        label = raw[len(prefix) :] if raw.startswith(prefix) else raw
        token_replacements.append(
            (label, "{" + key.replace("_id", "_label") + "}")
        )
    for key, raw in plan.get("layout", {}).items():
        if isinstance(raw, str) and raw.startswith("/"):
            replacements.append((raw, "{" + key + "}"))
    rendered = value
    for source, target in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        rendered = rendered.replace(source, target)
    # Bare participant/session labels appear as CLI values and path components.
    # Do not replace arbitrary substrings: a subject label such as ``pilot`` may
    # also occur in an unrelated parent directory or image path.
    for source, target in token_replacements:
        rendered = re.sub(
            rf"(^|[/=,:]){re.escape(source)}(?=$|[/,:])",
            lambda match, replacement=target: match.group(1) + replacement,
            rendered,
        )
    return rendered


def _normalized_artifacts(plan: dict[str, Any], task: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for requirement in task.get("artifacts", []):
        normalized = dict(requirement)
        normalized["patterns"] = [
            _normalize_value(str(pattern), plan, task)
            for pattern in requirement.get("patterns", [])
        ]
        result.append(normalized)
    return result


def executor_signature(plan: dict[str, Any], executor_id: str) -> str:
    signatures: set[bytes] = set()
    for task in plan.get("tasks", []):
        if task.get("executor_id") != executor_id:
            continue
        metadata = task.get("metadata", {})
        payload = {
            "executor_id": executor_id,
            "plugin_ids": sorted(task.get("plugin_ids", [])),
            "stages": sorted(task.get("stages", [])),
            "scope": task.get("scope"),
            "command": [
                _normalize_value(str(value), plan, task) for value in task.get("command", [])
            ],
            "environment": {
                key: _normalize_value(str(value), plan, task)
                for key, value in sorted(task.get("environment", {}).items())
            },
            "resource": {
                "class": task.get("resource_class"),
                "cpus": task.get("cpus"),
                "memory_gb": task.get("memory_gb"),
                "gpus": task.get("gpus"),
            },
            "required": task.get("required"),
            "artifacts": _normalized_artifacts(plan, task),
            "executor_metadata": {
                key: metadata.get(key)
                for key in (
                    "executor_version",
                    "executor_source",
                    "image_source",
                    "container_image",
                )
            },
        }
        signatures.add(_canonical_bytes(payload))
    if not signatures:
        raise EvidenceValidationError(f"target contains no tasks for executor {executor_id}")
    if len(signatures) != 1:
        raise EvidenceValidationError(
            f"executor {executor_id} has multiple command/resource/artifact signatures"
        )
    return _sha256_bytes(next(iter(signatures)))


def _image_path(plan: dict[str, Any], executor_id: str) -> Path:
    observed: set[str] = set()
    for task in plan.get("tasks", []):
        if task.get("executor_id") != executor_id:
            continue
        images = [str(value) for value in task.get("command", []) if str(value).endswith(".sif")]
        if len(images) != 1:
            raise EvidenceValidationError(
                f"executor {executor_id} must have exactly one image in every command"
            )
        observed.add(images[0])
    if len(observed) != 1:
        raise EvidenceValidationError(f"executor {executor_id} image paths disagree")
    return Path(observed.pop()).resolve(strict=True)


def protocol_signature(plan: dict[str, Any]) -> ProtocolSignature:
    acquisitions: list[dict[str, Any]] = []
    plugins: set[str] = set()
    formats: set[str] = set()
    for task in plan.get("tasks", []):
        plugins.update(str(value) for value in task.get("plugin_ids", []))
        if task.get("executor_id") != "neuroprep-bids-stage":
            continue
        metadata = task.get("metadata", {})
        source_format = str(metadata.get("source_format") or "unknown")
        formats.add(source_format)
        acquisitions.append(
            {
                "plugins": sorted(str(value) for value in task.get("plugin_ids", [])),
                "source_format": source_format,
                "bids_datatype": metadata.get("bids_datatype"),
                "bids_suffix": metadata.get("bids_suffix"),
                "source_metadata_sha256": _sha256_bytes(
                    _canonical_bytes(metadata.get("source_metadata", {}))
                ),
            }
        )
    payload = {
        "dataset_id": str(plan.get("dataset_id")),
        "acquisitions": sorted(acquisitions, key=lambda value: _canonical_bytes(value)),
        "modality_plugins": sorted(plugins),
        "source_formats": sorted(formats),
    }
    return ProtocolSignature(
        dataset_id=payload["dataset_id"],
        acquisition_count=len(acquisitions),
        modality_plugins=payload["modality_plugins"],
        source_formats=payload["source_formats"],
        signature_sha256=_sha256_bytes(_canonical_bytes(payload)),
    )


def _nonempty(path: Path) -> bool:
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(path.iterdir())
    return False


def _artifact_audit(
    plan: dict[str, Any], records: dict[str, dict[str, Any]], executor_id: str | None = None
) -> tuple[int, str]:
    inventory: list[dict[str, Any]] = []
    for task in plan.get("tasks", []):
        if executor_id is not None and task.get("executor_id") != executor_id:
            continue
        record = records.get(str(task.get("task_id")))
        if record is None:
            raise EvidenceValidationError(f"pilot state is missing task {task.get('task_id')}")
        if record["status"] != "complete" or record["return_code"] != 0:
            raise EvidenceValidationError(f"pilot task is not complete: {task.get('task_id')}")
        if record["fingerprint"] != task.get("fingerprint"):
            raise EvidenceValidationError(f"pilot task fingerprint mismatch: {task.get('task_id')}")
        if json.loads(record["task_json"]) != task:
            raise EvidenceValidationError(f"pilot task JSON mismatch: {task.get('task_id')}")
        if not record["log_path"]:
            raise EvidenceValidationError(f"pilot task log is absent: {task.get('task_id')}")
        log_path = Path(str(record["log_path"]))
        if not _nonempty(log_path):
            raise EvidenceValidationError(
                f"pilot task log is absent or empty: {task.get('task_id')}"
            )
        recorded = json.loads(record["artifacts_json"] or "[]")
        if not isinstance(recorded, list) or not recorded:
            raise EvidenceValidationError(
                f"pilot task recorded no artifacts: {task.get('task_id')}"
            )
        recorded_paths = {str(Path(value)) for value in recorded}
        matched_paths: set[str] = set()
        for requirement in task.get("artifacts", []):
            if requirement.get("validator", "exists_nonempty") != "exists_nonempty":
                raise EvidenceValidationError("unsupported artifact validator in pilot plan")
            matches: set[str] = set()
            for pattern in requirement.get("patterns", []):
                candidate = Path(pattern)
                expanded = str(
                    candidate
                    if candidate.is_absolute()
                    else Path(task["output_root"]) / candidate
                )
                matches.update(str(Path(value)) for value in glob.glob(expanded, recursive=True))
            matches = {value for value in matches if _nonempty(Path(value))}
            minimum = int(requirement.get("minimum_count", 1))
            if requirement.get("required", True) and len(matches) < minimum:
                raise EvidenceValidationError(
                    f"pilot artifact contract failed for {task.get('task_id')}"
                )
            matched_paths.update(matches)
        if not matched_paths or not matched_paths.issubset(recorded_paths):
            raise EvidenceValidationError(
                f"pilot artifacts disagree with state for {task.get('task_id')}"
            )
        for path in sorted(matched_paths):
            candidate = Path(path)
            inventory.append(
                {
                    "task_fingerprint": task["fingerprint"],
                    "artifact_name": candidate.name,
                    "size": candidate.stat().st_size if candidate.is_file() else -1,
                    "sha256": sha256_file(candidate) if candidate.is_file() else None,
                }
            )
    if not inventory:
        raise EvidenceValidationError("pilot artifact audit is empty")
    return len(inventory), _sha256_bytes(_canonical_bytes(inventory))


def _read_terminal_state(
    state_path: str | Path, plan: dict[str, Any], plan_path: str | Path
) -> tuple[dict[str, dict[str, Any]], dict[str, int], dict[str, str]]:
    state = Path(state_path).resolve(strict=True)
    connection = sqlite3.connect(state.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        rows = [dict(row) for row in connection.execute("SELECT * FROM tasks ORDER BY task_id")]
    finally:
        connection.close()
    if metadata.get("catalog_digest") != plan.get("catalog_digest"):
        raise EvidenceValidationError("pilot state catalog digest mismatch")
    if metadata.get("dataset_id") != plan.get("dataset_id"):
        raise EvidenceValidationError("pilot state dataset mismatch")
    if Path(metadata.get("plan_path", "")).resolve() != Path(plan_path).resolve():
        raise EvidenceValidationError("pilot state plan path mismatch")
    if len(rows) != len(plan.get("tasks", [])):
        raise EvidenceValidationError("pilot state task count differs from plan")
    counts = Counter(str(row["status"]) for row in rows)
    if counts != Counter({"complete": len(rows)}):
        raise EvidenceValidationError("pilot state is not terminal-success complete")
    return (
        {str(row["task_id"]): row for row in rows},
        dict(sorted(counts.items())),
        metadata,
    )


def _validate_release_index(
    path: str | Path,
    *,
    count_key: str,
    expected_count: int,
    dataset_id: str,
) -> tuple[FileIdentity, set[str]]:
    identity = _file_identity(path)
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise EvidenceValidationError(f"pilot {count_key} index is invalid")
    header = rows[0]
    if (
        header.get("record_type") != "index_metadata"
        or header.get("dataset_id") != dataset_id
        or header.get(count_key) != expected_count
        or len(rows) - 1 != expected_count
    ):
        raise EvidenceValidationError(f"pilot {count_key} index count mismatch")
    paths: set[str] = set()
    for row in rows[1:]:
        if row.get("dataset_id") != dataset_id:
            raise EvidenceValidationError(f"pilot {count_key} index dataset mismatch")
        artifact = Path(str(row.get("path") or "")).resolve(strict=True)
        if not _nonempty(artifact):
            raise EvidenceValidationError(f"pilot {count_key} index artifact is empty")
        if row.get("artifact_type") == "file":
            if not artifact.is_file() or artifact.stat().st_size != row.get("size"):
                raise EvidenceValidationError(
                    f"pilot {count_key} index file identity mismatch"
                )
        elif row.get("artifact_type") == "directory":
            child_count = sum(1 for item in artifact.rglob("*") if item.is_file())
            if not artifact.is_dir() or child_count != row.get("file_count"):
                raise EvidenceValidationError(
                    f"pilot {count_key} index directory identity mismatch"
                )
        else:
            raise EvidenceValidationError(f"pilot {count_key} index artifact type is invalid")
        paths.add(str(artifact))
    return identity, paths


def _validate_release(
    path: str | Path, plan: dict[str, Any], task_counts: dict[str, int]
) -> tuple[dict[str, Any], FileIdentity, FileIdentity, FileIdentity]:
    identity = _file_identity(path)
    release = _load_json(path)
    if release.get("ready") is not True:
        raise EvidenceValidationError("pilot release is not ready")
    if release.get("catalog_digest") != plan.get("catalog_digest"):
        raise EvidenceValidationError("pilot release catalog digest mismatch")
    if release.get("dataset_id") != plan.get("dataset_id"):
        raise EvidenceValidationError("pilot release dataset mismatch")
    if release.get("task_counts") != task_counts:
        raise EvidenceValidationError("pilot release task counts mismatch")
    indexes: dict[str, FileIdentity] = {}
    indexed_paths: dict[str, set[str]] = {}
    for count_key, index_key in (
        ("artifact_count", "artifact_index"),
        ("tensor_count", "tensor_index"),
    ):
        index_identity, paths = _validate_release_index(
            str(release.get(index_key, "")),
            count_key=count_key,
            expected_count=int(release.get(count_key, -1)),
            dataset_id=str(plan.get("dataset_id")),
        )
        indexes[index_key] = index_identity
        indexed_paths[index_key] = paths
    if not indexed_paths["tensor_index"] <= indexed_paths["artifact_index"]:
        raise EvidenceValidationError("pilot tensor index is not a subset of artifact index")
    return release, identity, indexes["artifact_index"], indexes["tensor_index"]


def _smoke_specs(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in plan.get("planned_smoke_tests", []):
        test_id = str(item.get("test_id") or "")
        executor_id = str(item.get("executor_id") or "")
        if not test_id or not executor_id or test_id in result:
            raise EvidenceValidationError("target smoke specifications are invalid")
        result[test_id] = item
    return result


def _unsigned_payload(document: BackendEvidenceDocument | dict[str, Any]) -> dict[str, Any]:
    if isinstance(document, BackendEvidenceDocument):
        payload = document.model_dump(mode="json")
    else:
        payload = dict(document)
    payload.pop("signature", None)
    return payload


def _sign_payload(payload: dict[str, Any], key: bytes, key_id: str) -> EvidenceSignature:
    if len(key) < 32:
        raise EvidenceValidationError("evidence signing key must contain at least 32 bytes")
    serialized = _canonical_bytes(payload)
    return EvidenceSignature(
        key_id=key_id,
        payload_sha256=_sha256_bytes(serialized),
        value=hmac.new(key, serialized, hashlib.sha256).hexdigest(),
    )


def _verify_signature(document: BackendEvidenceDocument, key: bytes) -> None:
    expected = _sign_payload(_unsigned_payload(document), key, document.signature.key_id)
    if not hmac.compare_digest(expected.payload_sha256, document.signature.payload_sha256):
        raise EvidenceValidationError("backend evidence payload digest mismatch")
    if not hmac.compare_digest(expected.value, document.signature.value):
        raise EvidenceValidationError("backend evidence signature mismatch")


def build_backend_evidence(
    *,
    pilot_plan_path: str | Path,
    target_plan_path: str | Path,
    pilot_state_path: str | Path,
    pilot_release_path: str | Path,
    git_repository: str | Path,
    signing_key: bytes,
    key_id: str,
) -> BackendEvidenceDocument:
    pilot_plan = _load_json(pilot_plan_path)
    target_plan = _load_json(target_plan_path)
    if pilot_plan.get("catalog_digest") != target_plan.get("catalog_digest"):
        raise EvidenceValidationError("pilot and target catalog digests differ")
    catalog_digest = str(pilot_plan.get("catalog_digest") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", catalog_digest):
        raise EvidenceValidationError("invalid catalog digest")
    target_git_commit = current_git_commit(git_repository)
    records, task_counts, state_metadata = _read_terminal_state(
        pilot_state_path, pilot_plan, pilot_plan_path
    )
    release, release_identity, artifact_index_identity, tensor_index_identity = (
        _validate_release(pilot_release_path, pilot_plan, task_counts)
    )
    total_artifacts, total_artifact_digest = _artifact_audit(pilot_plan, records)
    target_smokes = _smoke_specs(target_plan)
    if not target_smokes:
        raise EvidenceValidationError("target smoke intersection is empty")
    tests: list[BackendTestEvidence] = []
    image_cache: dict[Path, FileIdentity] = {}
    for test_id, smoke in sorted(target_smokes.items()):
        executor_id = str(smoke["executor_id"])
        if not any(
            task.get("executor_id") == executor_id for task in pilot_plan.get("tasks", [])
        ):
            continue
        pilot_signature = executor_signature(pilot_plan, executor_id)
        target_signature = executor_signature(target_plan, executor_id)
        if pilot_signature != target_signature:
            raise EvidenceValidationError(f"executor command mismatch for {executor_id}")
        pilot_image = _image_path(pilot_plan, executor_id)
        target_image = _image_path(target_plan, executor_id)
        if pilot_image != target_image:
            raise EvidenceValidationError(f"executor image mismatch for {executor_id}")
        if pilot_image not in image_cache:
            image_cache[pilot_image] = _file_identity(pilot_image)
        artifact_count, artifact_digest = _artifact_audit(
            pilot_plan, records, executor_id
        )
        pilot_task_count = sum(
            task.get("executor_id") == executor_id for task in pilot_plan.get("tasks", [])
        )
        tests.append(
            BackendTestEvidence(
                test_id=test_id,
                capability=str(smoke.get("capability") or ""),
                package_id=str(smoke.get("package_id") or ""),
                executor_id=executor_id,
                status="passed",
                executor_signature_sha256=pilot_signature,
                pilot_task_count=pilot_task_count,
                artifact_count=artifact_count,
                artifact_audit_sha256=artifact_digest,
                image=image_cache[pilot_image],
            )
        )
    if not tests:
        raise EvidenceValidationError("target smoke intersection is empty")
    target_protocol = protocol_signature(target_plan)
    pilot = PilotAudit(
        dataset_id=str(pilot_plan["dataset_id"]),
        plan=_file_identity(pilot_plan_path),
        state=_file_identity(pilot_state_path),
        release_summary=release_identity,
        artifact_index=artifact_index_identity,
        tensor_index=tensor_index_identity,
        release_ready=True,
        task_counts=task_counts,
        artifact_count=total_artifacts,
        tensor_count=int(release["tensor_count"]),
        artifact_audit_sha256=total_artifact_digest,
        protocol=protocol_signature(pilot_plan),
        git=_pilot_git_provenance(pilot_plan, state_metadata, release),
    )
    target_test_specs = [
        SmokeTestIdentity(
            test_id=test_id,
            capability=str(smoke.get("capability") or ""),
            package_id=str(smoke.get("package_id") or ""),
            executor_id=str(smoke["executor_id"]),
        )
        for test_id, smoke in sorted(target_smokes.items())
    ]
    unsigned = {
        "schema_version": "2.0",
        "generated_at": str(release["generated_at"]),
        "catalog_digest": catalog_digest,
        "target_git_commit": target_git_commit,
        "pilot": pilot.model_dump(mode="json"),
        "target": TargetBinding(
            dataset_id=str(target_plan["dataset_id"]),
            qualification_plan=_file_identity(target_plan_path),
            protocol=target_protocol,
            smoke_test_ids=[item.test_id for item in target_test_specs],
            smoke_tests=target_test_specs,
        ).model_dump(mode="json"),
        "tests": [item.model_dump(mode="json") for item in tests],
    }
    return BackendEvidenceDocument.model_validate(
        {**unsigned, "signature": _sign_payload(unsigned, signing_key, key_id)}
    )


def evidence_document_sha256(document: BackendEvidenceDocument) -> str:
    return _sha256_bytes(_canonical_bytes(document))


def validate_backend_evidence(
    *,
    evidence_path: str | Path,
    signing_key: bytes | None,
    target_plan: dict[str, Any],
    target_smoke_tests: list[dict[str, Any]],
    catalog_digest: str,
    target_git_commit: str,
) -> LoadedBackendEvidence:
    candidate = validated_cluster_path(str(evidence_path), must_exist=True).resolve()
    loaded = _load_json(candidate)
    required = _smoke_specs({"planned_smoke_tests": target_smoke_tests})
    if not loaded:
        raise EvidenceValidationError("backend evidence cannot be empty")
    legacy = all(value in {"passed", "failed"} for value in loaded.values())
    if legacy:
        statuses = {
            str(key): value for key, value in loaded.items() if str(key) in required
        }
        binding = BackendEvidenceBinding(
            trust="legacy_untrusted",
            source_path=str(candidate),
            document_sha256=sha256_file(candidate),
            required_test_ids=sorted(required),
            matched_test_ids=sorted(statuses),
        )
        return LoadedBackendEvidence(statuses=statuses, binding=binding)
    if signing_key is None:
        raise EvidenceValidationError("rich backend evidence requires a signing key")
    document = BackendEvidenceDocument.model_validate(loaded)
    _verify_signature(document, signing_key)
    if (
        document.catalog_digest != catalog_digest
        or target_plan.get("catalog_digest") != catalog_digest
    ):
        raise EvidenceValidationError("backend evidence catalog digest mismatch")
    if document.target_git_commit != target_git_commit:
        raise EvidenceValidationError("backend evidence target git commit mismatch")
    target_protocol = protocol_signature(target_plan)
    if document.target.dataset_id != target_plan.get("dataset_id"):
        raise EvidenceValidationError("backend evidence target dataset mismatch")
    if document.target.protocol.signature_sha256 != target_protocol.signature_sha256:
        raise EvidenceValidationError("backend evidence target protocol mismatch")
    signed_smoke_tests = {
        item.test_id: item.model_dump(mode="json")
        for item in document.target.smoke_tests
    }
    current_smoke_tests = {
        test_id: SmokeTestIdentity(
            test_id=test_id,
            capability=str(item.get("capability") or ""),
            package_id=str(item.get("package_id") or ""),
            executor_id=str(item["executor_id"]),
        ).model_dump(mode="json")
        for test_id, item in required.items()
    }
    if signed_smoke_tests != current_smoke_tests:
        raise EvidenceValidationError("backend evidence target smoke tests mismatch")
    statuses: dict[str, Literal["passed", "failed"]] = {}
    for item in document.tests:
        smoke = required.get(item.test_id)
        if smoke is None:
            raise EvidenceValidationError(
                "backend evidence is not an exact target-smoke intersection"
            )
        if str(smoke["executor_id"]) != item.executor_id:
            raise EvidenceValidationError(f"backend evidence executor mismatch: {item.test_id}")
        if executor_signature(target_plan, item.executor_id) != item.executor_signature_sha256:
            raise EvidenceValidationError(f"backend evidence command mismatch: {item.test_id}")
        image = _image_path(target_plan, item.executor_id)
        if str(image) != item.image.path:
            raise EvidenceValidationError(f"backend evidence image mismatch: {item.test_id}")
        if image.stat().st_size != item.image.size or sha256_file(image) != item.image.sha256:
            raise EvidenceValidationError(f"backend evidence image hash mismatch: {item.test_id}")
        statuses[item.test_id] = item.status
    if not statuses:
        raise EvidenceValidationError("target smoke intersection is empty")
    binding = BackendEvidenceBinding(
        trust="verified",
        source_path=str(candidate),
        document_sha256=evidence_document_sha256(document),
        catalog_digest=document.catalog_digest,
        pilot_git_commit=document.pilot.git.commit,
        target_git_commit=document.target_git_commit,
        key_id=document.signature.key_id,
        pilot_plan_sha256=document.pilot.plan.sha256,
        target_protocol_sha256=document.target.protocol.signature_sha256,
        required_test_ids=sorted(required),
        matched_test_ids=sorted(
            test_id for test_id, status in statuses.items() if status == "passed"
        ),
    )
    return LoadedBackendEvidence(statuses=statuses, binding=binding)


def evidence_allows_execution(
    planned_smoke_tests: list[dict[str, Any]], binding: BackendEvidenceBinding | None
) -> bool:
    required = set(binding.required_test_ids) if binding else set()
    required.update(
        str(item.get("test_id")) for item in planned_smoke_tests if item.get("test_id")
    )
    if not required:
        return True
    if binding is None or binding.trust != "verified":
        return False
    return required <= set(binding.matched_test_ids)


def revalidate_execution_evidence(
    *,
    plan: dict[str, Any],
    signing_key: bytes | None,
    target_git_commit: str,
) -> LoadedBackendEvidence | None:
    """Re-open signed evidence before a separately invoked run initialization."""

    qualification_tests = list(
        plan.get("backend_qualification_tests") or plan.get("planned_smoke_tests") or []
    )
    binding_value = plan.get("backend_evidence")
    binding = (
        BackendEvidenceBinding.model_validate(binding_value) if binding_value else None
    )
    if not qualification_tests:
        return None
    if binding is None or binding.trust != "verified":
        raise EvidenceValidationError(
            "execution requires verified signed backend evidence"
        )
    if signing_key is None:
        raise EvidenceValidationError(
            "execution evidence revalidation requires the signing key"
        )
    loaded = validate_backend_evidence(
        evidence_path=binding.source_path,
        signing_key=signing_key,
        target_plan=plan,
        target_smoke_tests=qualification_tests,
        catalog_digest=str(plan.get("catalog_digest") or ""),
        target_git_commit=target_git_commit,
    )
    if loaded.binding != binding:
        raise EvidenceValidationError(
            "embedded backend evidence provenance differs from its signed source"
        )
    if not evidence_allows_execution(qualification_tests, loaded.binding):
        raise EvidenceValidationError(
            "signed backend evidence does not pass every qualification test"
        )
    return loaded


def write_evidence_document(document: BackendEvidenceDocument, output: str | Path) -> Path:
    destination = validated_cluster_path(str(output))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite backend evidence: {destination}")
    rendered = json.dumps(document.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    return destination
