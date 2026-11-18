from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from .backend_evidence import (
    current_git_commit,
    evidence_allows_execution,
    read_signing_key,
    revalidate_execution_evidence,
    validate_backend_evidence,
)
from .compiler import compile_execution_plan, save_execution_plan
from .execution_models import RunLayout
from .executors import builtin_executor_catalog
from .images import image_inventory
from .models import DatasetSpec, SourceRecord
from .openjev import OpenJevClient, review_acquisitions
from .policy import (
    HardwareProfile,
    RunPolicy,
    compile_prompt_policy,
    detect_hardware_profile,
    plan_backends,
)
from .preflight import preflight_execution_plan
from .registry import builtin_registry
from .runtime_assets import prepare_runtime_assets
from .services import ManifestDocument, build_graph_and_plan, load_manifest
from .settings import ALLOWED_ROOTS, validated_agentic_project, validated_cluster_path
from .state import RunState

_PATH_TOKEN = (
    rf"(?:{'|'.join(re.escape(str(root)) for root in ALLOWED_ROOTS)})"
    r"(?:/[^\s\"'<>]*)?"
)
_PATH_RE = re.compile(_PATH_TOKEN)
_SOURCE_RE = re.compile(
    rf"(?:data|input|source)(?:\s+(?:is|are|located))?"
    rf"(?:\s+(?:at|in|under|from))?\s+(?P<path>{_PATH_TOKEN})",
    re.IGNORECASE,
)
_OUTPUT_RE = re.compile(
    rf"(?:save|write|store|output)(?:[^.\n]{{0,240}}?)"
    rf"(?:to|under|in|at)\s+(?P<path>{_PATH_TOKEN})",
    re.IGNORECASE,
)
_TRAILING_PATH_PUNCTUATION = ".,;:!?)]}"
_SKIP_DIRECTORIES = {
    ".git",
    ".cache",
    "__pycache__",
    "cache",
    "code",
    "derivatives",
    "logs",
    "work",
}
_IMAGE_SUFFIXES = (
    ".nii.gz",
    ".ome.tif",
    ".nii",
    ".ecat",
    ".tiff",
    ".tif",
    ".mp4",
)
_SENSOR_SUFFIXES = (
    ".edf",
    ".bdf",
    ".vhdr",
    ".set",
    ".fif",
    ".snirf",
    ".con",
    ".sqd",
)
_MRS_SUFFIXES = (".rda", ".spar", ".sdat", ".7")
_TABULAR_SUFFIXES = (".tsv.gz", ".tsv", ".csv")
_TABULAR_TERMS = (
    "assessment",
    "behavior",
    "cognitive",
    "ecg",
    "events",
    "genetic",
    "genomic",
    "physio",
    "pulse",
    "respiratory",
    "snp",
)
_GENETIC_SUFFIXES = (".vcf", ".bcf")


class PromptPathBinding(BaseModel):
    """Deterministic, non-LLM execution roots extracted from a request."""

    model_config = ConfigDict(extra="forbid")

    source_root: str
    output_root: str

    @model_validator(mode="after")
    def safe_separated_roots(self) -> PromptPathBinding:
        source = validated_cluster_path(self.source_root, must_exist=True).resolve()
        output = validated_cluster_path(self.output_root).resolve()
        if not source.is_dir():
            raise ValueError("prompt source root must be an existing directory")
        if output in ALLOWED_ROOTS:
            raise ValueError("prompt output root cannot be an approved storage root itself")
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("prompt source and output roots must not overlap")
        self.source_root = str(source)
        self.output_root = str(output)
        return self


def _clean_extracted_path(value: str) -> str:
    return value.rstrip(_TRAILING_PATH_PUNCTUATION)


def _extract_marked_path(pattern: re.Pattern[str], request: str) -> str | None:
    matches = list(pattern.finditer(request))
    if not matches:
        return None
    return _clean_extracted_path(matches[-1].group("path"))


def _compared_executor_ids(smoke_tests: list[dict[str, Any]]) -> set[str]:
    """Executors of every capability with two or more backends to qualify."""
    by_capability: dict[str, set[str]] = {}
    for item in smoke_tests:
        by_capability.setdefault(str(item["capability"]), set()).add(str(item["executor_id"]))
    return {executor for group in by_capability.values() if len(group) > 1 for executor in group}


def resolve_prompt_paths(
    request: str,
    *,
    source_root: str | None = None,
    output_root: str | None = None,
) -> PromptPathBinding:
    """Resolve X and Y without granting the LLM authority over paths."""

    detected_source = _extract_marked_path(_SOURCE_RE, request)
    detected_output = _extract_marked_path(_OUTPUT_RE, request)
    mentioned = [_clean_extracted_path(match.group(0)) for match in _PATH_RE.finditer(request)]
    mentioned = list(dict.fromkeys(mentioned))

    if detected_source is None and len(mentioned) >= 2:
        detected_source = mentioned[0]
    if detected_output is None and len(mentioned) >= 2:
        detected_output = mentioned[-1]
    if detected_source is None and source_root is None and len(mentioned) == 1:
        detected_source = mentioned[0]

    if (
        source_root
        and detected_source
        and Path(source_root).resolve() != Path(detected_source).resolve()
    ):
        raise ValueError("explicit source root conflicts with the prompt")
    if (
        output_root
        and detected_output
        and (Path(output_root).resolve() != Path(detected_output).resolve())
    ):
        raise ValueError("explicit output root conflicts with the prompt")

    selected_source = source_root or detected_source
    selected_output = output_root or detected_output
    if not selected_source:
        raise ValueError(
            "could not resolve a source root; state 'data is at /approved/path' "
            "or pass --source-root"
        )
    if not selected_output:
        raise ValueError(
            "could not resolve an output root; state 'save to /approved/path' or pass --output-root"
        )
    return PromptPathBinding(source_root=selected_source, output_root=selected_output)


def _suffix(path: Path) -> str:
    lowered = path.name.lower()
    for suffix in (
        ".nii.gz",
        ".tsv.gz",
        ".ome.tif",
        *_SENSOR_SUFFIXES,
        *_MRS_SUFFIXES,
        *_GENETIC_SUFFIXES,
    ):
        if lowered.endswith(suffix):
            return suffix
    return path.suffix.lower()


def _looks_like_dicom(path: Path) -> bool:
    if path.suffix.lower() == ".dcm":
        return True
    if path.suffix:
        return False
    try:
        if path.stat().st_size < 132:
            return False
        with path.open("rb") as stream:
            stream.seek(128)
            return stream.read(4) == b"DICM"
    except OSError:
        return False


def _is_primary_file(path: Path) -> bool:
    suffix = _suffix(path)
    lowered = path.name.lower()
    if suffix in _IMAGE_SUFFIXES or suffix in _SENSOR_SUFFIXES:
        return True
    if suffix in _MRS_SUFFIXES:
        return True
    if suffix == ".dat" and any(term in lowered for term in ("mrs", "spectro", "svs")):
        return True
    if suffix in _GENETIC_SUFFIXES:
        return True
    if suffix in _TABULAR_SUFFIXES:
        return any(term in lowered for term in _TABULAR_TERMS)
    return False


def _source_items(source_root: Path, max_records: int) -> tuple[list[Path], dict[str, int]]:
    items: list[Path] = []
    ignored_files = 0
    dicom_directories = 0
    directory_modalities = 0
    for current, directories, files in os.walk(source_root, followlinks=False):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in sorted(directories):
            if name.startswith(".") or name.lower() in _SKIP_DIRECTORIES:
                continue
            candidate = current_path / name
            if candidate.name.lower().endswith(".ds"):
                items.append(candidate)
                directory_modalities += 1
            else:
                kept_directories.append(name)
        directories[:] = kept_directories

        file_paths = [current_path / name for name in sorted(files)]
        dicom_files = [path for path in file_paths if _looks_like_dicom(path)]
        if dicom_files:
            items.append(current_path)
            dicom_directories += 1
        for path in file_paths:
            if path in dicom_files:
                continue
            if _is_primary_file(path):
                items.append(path)
            else:
                ignored_files += 1
        if len(items) > max_records:
            raise ValueError(f"source inventory exceeds {max_records} acquisitions")

    unique = sorted({path.resolve() for path in items}, key=str)
    if not unique:
        raise ValueError("source inventory found no supported acquisitions")
    return unique, {
        "records": len(unique),
        "ignored_files": ignored_files,
        "dicom_directories": dicom_directories,
        "directory_modalities": directory_modalities,
    }


def _sidecar_path(path: Path) -> Path | None:
    if path.is_dir():
        return None
    if path.name.lower().endswith(".nii.gz"):
        return path.with_name(path.name[:-7] + ".json")
    return path.with_suffix(".json")


def _metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    sidecar = _sidecar_path(path)
    if sidecar is not None and sidecar.is_file():
        if sidecar.stat().st_size > 16 * 1024 * 1024:
            raise ValueError(f"sidecar exceeds 16 MiB: {sidecar}")
        try:
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON sidecar: {sidecar}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"JSON sidecar is not an object: {sidecar}")
        metadata.update(loaded)

    searchable = str(path).lower()
    suffix = _suffix(path)
    if "Modality" not in metadata:
        if suffix in {".edf", ".bdf", ".vhdr", ".set"}:
            metadata["Modality"] = "EEG"
        elif suffix in {".fif", ".con", ".sqd"} or path.name.lower().endswith(".ds"):
            metadata["Modality"] = "MEG"
        elif "_pet" in searchable or " pet" in searchable:
            metadata["Modality"] = "PT"
        elif re.search(r"(?<![a-z])ct(?![a-z])", searchable):
            metadata["Modality"] = "CT"
        elif suffix in _IMAGE_SUFFIXES or suffix in _MRS_SUFFIXES or path.is_dir():
            metadata["Modality"] = "MR"
    metadata.setdefault("SeriesDescription", path.name)
    metadata.setdefault("NeuroPrepInventorySource", "prompt-run-deterministic-v1")
    return metadata


def _subject_and_session(source_root: Path, path: Path) -> tuple[str, str]:
    relative = path.relative_to(source_root).as_posix()
    subject = re.search(r"(?:^|[/_])sub-([A-Za-z0-9]+)", relative, re.IGNORECASE)
    session = re.search(r"(?:^|[/_])ses(?:s)?-([A-Za-z0-9]+)", relative, re.IGNORECASE)
    if subject is None:
        subject = re.search(r"(?:^|[/_])(OAS\d+|P\d+|RID\d+)(?:[/_]|$)", relative, re.I)
    if session is None:
        session = re.search(r"(?:^|[_/])(d-?\d+|v\d+)(?:[_/]|$)", relative, re.I)

    parts = [part for part in Path(relative).parts[:-1] if part]
    subject_value = subject.group(1) if subject else (parts[0] if parts else "dataset")
    session_value = session.group(1) if session else (parts[1] if len(parts) > 1 else "baseline")
    subject_value = re.sub(r"[^A-Za-z0-9]+", "", subject_value) or "dataset"
    session_value = re.sub(r"[^A-Za-z0-9-]+", "", session_value) or "baseline"
    return subject_value, session_value


def _days_from_session(session: str) -> int | None:
    match = re.fullmatch(r"d(-?\d+)", session, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _acquisition_time(metadata: dict[str, Any]) -> datetime | None:
    value = metadata.get("AcquisitionDateTime")
    if isinstance(value, str):
        normalized = value.strip().replace("Z", "+00:00")
        for candidate in (normalized, normalized.replace(".", "")):
            try:
                return datetime.fromisoformat(candidate)
            except ValueError:
                pass
    date = metadata.get("AcquisitionDate") or metadata.get("StudyDate")
    time = metadata.get("AcquisitionTime") or metadata.get("SeriesTime")
    if isinstance(date, str):
        compact_date = re.sub(r"[^0-9]", "", date)[:8]
        compact_time = re.sub(r"[^0-9.]", "", str(time or "000000"))
        if len(compact_date) == 8:
            try:
                return datetime.strptime(
                    compact_date + compact_time.split(".", 1)[0].ljust(6, "0")[:6],
                    "%Y%m%d%H%M%S",
                )
            except ValueError:
                return None
    return None


def _file_format(path: Path) -> str:
    if path.is_dir():
        return "DICOM" if not path.name.lower().endswith(".ds") else "MEG"
    suffix = _suffix(path)
    if suffix in {".nii", ".nii.gz"}:
        return "NIFTI"
    if suffix in {".edf", ".bdf", ".vhdr", ".set", ".fif", ".con", ".sqd"}:
        return "ELECTROPHYSIOLOGY"
    if suffix in _MRS_SUFFIXES or suffix == ".dat":
        return "MRS"
    return suffix.lstrip(".").upper() or "UNKNOWN"


def infer_dataset_id(source_root: str) -> str:
    name = Path(source_root).name.lower()
    normalized = re.sub(r"[^a-z0-9_-]+", "-", name).strip("-_")
    if not normalized:
        normalized = "prompt-dataset"
    return normalized[:80]


def infer_adapter_id(source_root: str, dataset_id: str) -> str:
    registry = builtin_registry()
    direct = registry.dataset_adapter(dataset_id)
    if direct.adapter_id != "generic":
        return direct.adapter_id
    haystack = re.sub(r"[^a-z0-9]+", " ", source_root.lower())
    for adapter in registry.datasets.values():
        if adapter.adapter_id == "generic":
            continue
        names = (adapter.adapter_id, *adapter.aliases)
        for name in names:
            needle = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
            if needle and re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack):
                return adapter.adapter_id
    return "generic"


def discover_source_manifest(
    source_root: str,
    *,
    dataset_id: str | None = None,
    adapter_id: str | None = None,
    max_records: int = 100_000,
) -> tuple[ManifestDocument, dict[str, Any]]:
    root = validated_cluster_path(source_root, must_exist=True).resolve()
    selected_dataset = dataset_id or infer_dataset_id(str(root))
    selected_adapter = adapter_id or infer_adapter_id(str(root), selected_dataset)
    items, report = _source_items(root, max_records)
    records: list[SourceRecord] = []
    for path in items:
        metadata = _metadata(path)
        subject, session = _subject_and_session(root, path)
        records.append(
            SourceRecord(
                dataset_id=selected_dataset,
                source_path=str(path),
                source_subject_id=subject,
                source_session_id=session,
                acquisition_time=_acquisition_time(metadata),
                days_from_baseline=_days_from_session(session),
                file_format=_file_format(path),
                metadata=metadata,
            )
        )
    manifest = ManifestDocument(
        dataset=DatasetSpec(
            dataset_id=selected_dataset,
            adapter_id=selected_adapter,
            source_roots=[str(root)],
            metadata={
                "inventory": "prompt-run-deterministic-v1",
                "inventory_record_count": len(records),
            },
        ),
        records=records,
    )
    return manifest, {
        **report,
        "dataset_id": selected_dataset,
        "adapter_id": selected_adapter,
        "source_root": str(root),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_manifest_sources(manifest: ManifestDocument, source_root: Path) -> None:
    for record in manifest.records:
        candidate = validated_cluster_path(record.source_path, must_exist=True).resolve()
        if candidate != source_root and source_root not in candidate.parents:
            raise ValueError(f"manifest source is outside the prompt source root: {candidate}")


def _identity_key(output_root: Path, supplied: str | None) -> tuple[str, str]:
    if supplied is not None:
        if len(supplied) < 12:
            raise ValueError("identity key must contain at least 12 characters")
        return supplied, "supplied"
    path = output_root / "state" / "identity.key"
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(key + "\n")
    return key, "generated"


def _submit_workers(
    plan_path: Path,
    gpu_workers: int,
    agentic_project: str,
    backend_evidence_key: str | None = None,
) -> dict[str, str]:
    launcher = Path(agentic_project) / "scripts/launch_neuroprep_run.sh"
    environment = os.environ.copy()
    environment["NEUROPREP_AGENTIC_PROJECT"] = agentic_project
    if backend_evidence_key is not None:
        environment["NEUROPREP_BACKEND_EVIDENCE_KEY"] = backend_evidence_key
    completed = subprocess.run(
        [str(launcher), str(plan_path), str(gpu_workers)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Slurm launch failed: " + (completed.stderr.strip() or completed.stdout.strip())
        )
    parsed: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            parsed[key.strip()] = value.strip()
    if parsed.get("launch_complete") != "true":
        raise RuntimeError("Slurm launcher did not report launch_complete=true")
    return parsed


def prepare_prompt_run(
    request: str,
    *,
    source_root: str | None = None,
    output_root: str | None = None,
    manifest_path: str | None = None,
    dataset_id: str | None = None,
    adapter_id: str | None = None,
    identity_key: str | None = None,
    agentic_project: str | None = None,
    hardware_profile: HardwareProfile | None = None,
    backend_evidence: str | None = None,
    backend_evidence_key: str | None = None,
    freesurfer_license: str | None = None,
    execute_optional: bool = False,
    allow_quarantine: bool = False,
    allow_unsupported: bool = False,
    initialize: bool = False,
    submit: bool = False,
    gpu_workers: int = 2,
    openjev_review: bool = False,
    policy: RunPolicy | None = None,
    bound_roots: bool = False,
    qualify_alternatives: bool = False,
) -> dict[str, Any]:
    if not request.strip():
        raise ValueError("prompt request cannot be empty")
    if gpu_workers < 1 or gpu_workers > 32:
        raise ValueError("gpu_workers must be between 1 and 32")
    resolved_agentic_project = validated_agentic_project(agentic_project)
    # bound_roots: the caller (the autopilot) already bound the prompt's paths and stages the
    # probe and pilot elsewhere; PromptPathBinding still enforces the allowed roots.
    binding = (
        PromptPathBinding(source_root=source_root, output_root=output_root)
        if bound_roots and source_root and output_root
        else resolve_prompt_paths(request, source_root=source_root, output_root=output_root)
    )
    source = Path(binding.source_root)
    output = Path(binding.output_root)
    if output.exists() and any(output.iterdir()):
        raise ValueError("prompt output root already exists and is not empty")
    output.mkdir(parents=True, exist_ok=True)

    request_path = output / "request.txt"
    request_path.write_text(request.strip() + "\n", encoding="utf-8")
    profile = hardware_profile or detect_hardware_profile()
    hardware_path = output / "hardware.json"
    _write_json(hardware_path, profile)
    # A caller (the autopilot) may pass an already-compiled policy so that probe, pilot and
    # production runs share one model decision instead of sampling the model repeatedly.
    policy = policy or compile_prompt_policy(request, profile)
    policy_path = output / "policy.json"
    _write_json(policy_path, policy)
    if manifest_path:
        manifest = load_manifest(manifest_path)
        _validate_manifest_sources(manifest, source)
        inventory_report: dict[str, Any] = {
            "mode": "supplied_manifest",
            "manifest_source": str(validated_cluster_path(manifest_path, must_exist=True)),
            "record_count": len(manifest.records),
            "dataset_id": manifest.dataset.dataset_id,
            "adapter_id": manifest.dataset.adapter_id,
            "source_root": str(source),
        }
    else:
        manifest, inventory_report = discover_source_manifest(
            str(source), dataset_id=dataset_id, adapter_id=adapter_id
        )
    if not manifest.records:
        raise ValueError("source inventory produced no supported input records")

    manifest_destination = output / "manifest.json"
    inventory_path = output / "inventory.json"
    _write_json(manifest_destination, manifest)
    _write_json(inventory_path, inventory_report)

    key, key_source = _identity_key(output, identity_key)
    graph, logical_plan = build_graph_and_plan(manifest, key)
    graph_path = output / "acquisition-graph.json"
    logical_plan_path = output / "logical-plan.json"
    _write_json(graph_path, graph)
    _write_json(logical_plan_path, logical_plan)
    openjev_path: Path | None = None
    if openjev_review:
        unresolved = [
            item.model_dump(mode="json")
            for item in (*graph.quarantined, *graph.unsupported)
        ]
        review = review_acquisitions(unresolved, OpenJevClient()) if unresolved else []
        openjev_path = output / "openjev-review.json"
        _write_json(
            openjev_path,
            {
                "schema_version": "1.0",
                "advisory_only": True,
                "reviewed_count": len(review),
                "decisions": review,
            },
        )

    if graph.quarantined and not allow_quarantine:
        raise ValueError(
            f"{len(graph.quarantined)} acquisitions are quarantined; review before execution"
        )
    if graph.unsupported and not allow_unsupported:
        raise ValueError(
            f"{len(graph.unsupported)} acquisitions are unsupported; add a plugin first"
        )

    registry = builtin_registry()
    catalog = builtin_executor_catalog()
    missing = catalog.audit(registry)
    if missing:
        raise ValueError("executor catalog gaps: " + ", ".join(missing))
    layout = RunLayout(
        dataset_id=manifest.dataset.dataset_id,
        source_root=str(source),
        bids_root=str(output / "bids"),
        derivatives_root=str(output / "derivatives"),
        work_root=str(output / "work"),
        state_path=str(output / "state" / "tasks.sqlite"),
        agentic_project=str(resolved_agentic_project),
        freesurfer_license=freesurfer_license,
        execute_optional=execute_optional,
    )
    probe_plan = compile_execution_plan(
        graph,
        logical_plan,
        layout,
        catalog,
        run_policy=policy.model_dump(mode="json"),
        policy_digest=policy.digest(),
        backend_decisions=[],
        planned_smoke_tests=[],
    )
    required_executor_ids = {task.executor_id for task in probe_plan.tasks}
    unqualified_backend_plan = plan_backends(
        policy,
        profile,
        {},
        required_executor_ids=required_executor_ids,
        required_stages={stage for task in probe_plan.tasks for stage in task.stages},
        mandatory_stages={task.stage for task in logical_plan.tasks if task.required},
    )
    target_smoke_tests = [
        item.model_dump(mode="json") for item in unqualified_backend_plan.smoke_tests
    ]
    evidence: dict[str, str] = {}
    runtimes: dict[str, float] = {}
    evidence_binding = None
    signing_key: bytes | None = None
    if backend_evidence:
        signing_key = read_signing_key(backend_evidence_key) if backend_evidence_key else None
        loaded_evidence = validate_backend_evidence(
            evidence_path=backend_evidence,
            signing_key=signing_key,
            target_plan=probe_plan.model_dump(mode="json"),
            target_smoke_tests=target_smoke_tests,
            catalog_digest=catalog.digest(),
            target_git_commit=current_git_commit(layout.agentic_project),
        )
        evidence = dict(loaded_evidence.statuses)
        runtimes = dict(loaded_evidence.runtimes)
        evidence_binding = loaded_evidence.binding
    backend_plan = plan_backends(
        policy,
        profile,
        evidence,
        required_executor_ids=required_executor_ids,
        required_stages={stage for task in probe_plan.tasks for stage in task.stages},
        mandatory_stages={task.stage for task in logical_plan.tasks if task.required},
        backend_evidence=evidence_binding,
        runtimes=runtimes,
    )
    backend_path = output / "backend-plan.json"
    _write_json(backend_path, backend_plan)
    if not backend_plan.ready:
        raise ValueError("no integrated hardware-compatible backend for every required capability")
    should_initialize = initialize or submit
    selected_executor_ids = {
        item.selected_executor_id for item in backend_plan.decisions if item.selected_executor_id
    }
    if should_initialize and not evidence_allows_execution(
        target_smoke_tests, evidence_binding, selected_executor_ids
    ):
        raise ValueError(
            "initialization requires signed, target-bound backend evidence for every "
            "required smoke test"
        )
    if should_initialize and backend_plan.smoke_tests:
        raise ValueError(
            f"{len(backend_plan.smoke_tests)} required backend smoke tests still require evidence; "
            "run bounded qualification tasks and pass signed --backend-evidence"
        )
    plan = compile_execution_plan(
        graph,
        logical_plan,
        layout,
        catalog,
        run_policy=policy.model_dump(mode="json"),
        policy_digest=policy.digest(),
        backend_decisions=[item.model_dump(mode="json") for item in backend_plan.decisions],
        backend_qualification_tests=target_smoke_tests,
        planned_smoke_tests=[item.model_dump(mode="json") for item in backend_plan.smoke_tests],
        backend_evidence=evidence_binding,
        qualification_candidates=(
            _compared_executor_ids(target_smoke_tests) if qualify_alternatives else ()
        ),
    )
    if should_initialize and evidence_binding is not None:
        # The routed plan may run an alternative the default-routed target did not:
        # re-check every executor it runs against its signed pilot command and image.
        revalidate_execution_evidence(
            plan=plan.model_dump(mode="json"),
            signing_key=signing_key,
            target_git_commit=current_git_commit(layout.agentic_project),
        )
    plan_path = save_execution_plan(plan, str(output / "execution-plan.json"))
    asset_report = prepare_runtime_assets(plan)
    asset_path = output / "runtime-assets.json"
    _write_json(asset_path, asset_report)
    images_report = image_inventory(plan, catalog)
    images_path = output / "images.json"
    _write_json(images_path, images_report)
    preflight = preflight_execution_plan(plan, catalog)
    preflight_path = output / "preflight.json"
    _write_json(preflight_path, preflight)

    if should_initialize and not preflight.ready:
        raise ValueError("execution preflight failed; refusing to initialize or submit")
    state_summary: dict[str, Any] | None = None
    if should_initialize:
        state = RunState(plan.layout.state_path)
        state.initialize(plan, str(plan_path))
        state_summary = state.summary()
    launch = (
        _submit_workers(
            plan_path,
            gpu_workers,
            plan.layout.agentic_project,
            backend_evidence_key=backend_evidence_key,
        )
        if submit
        else None
    )
    result = {
        "ready": preflight.ready,
        "submitted": submit,
        "binding": binding.model_dump(mode="json"),
        "dataset_id": manifest.dataset.dataset_id,
        "adapter_id": manifest.dataset.adapter_id,
        "record_count": len(manifest.records),
        "task_count": len(plan.tasks),
        "quarantined_count": len(graph.quarantined),
        "unsupported_count": len(graph.unsupported),
        "identity_key_source": key_source,
        "execution_plan": str(plan_path),
        "state_path": plan.layout.state_path,
        "preflight": str(preflight_path),
        "state_summary": state_summary,
        "launch": launch,
        "backend_evidence": (
            evidence_binding.model_dump(mode="json") if evidence_binding else None
        ),
        "openjev_review": str(openjev_path) if openjev_path else None,
    }
    _write_json(output / "prompt-run.json", result)
    return result
