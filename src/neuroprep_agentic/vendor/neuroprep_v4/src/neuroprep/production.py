from __future__ import annotations

import csv
import hashlib
import json
import resource
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .engine import TaskDatabase


DEFAULT_STORAGE_GB = {
    "smriprep": 2.5,
    "deepprep": 10.0,
    "deepprep-postprocess": 8.0,
    "neuroprep-structural": 1.0,
    "fmriprep": 12.0,
    "xcpd": 8.0,
    "qsiprep": 15.0,
    "petprep": 4.0,
    "aslprep": 4.0,
    "qsmxt": 6.0,
    "neuroprep-ct": 1.5,
    "neuroprep-mra": 1.5,
    "mne-bids-pipeline-eeg": 3.0,
    "mne-bids-pipeline-meg": 12.0,
    "mne-bids-pipeline-ieeg": 8.0,
}


def verify_container_lock(lock_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Container lock must contain an object")
    tools = payload.get("tools") if isinstance(payload.get("tools"), dict) else {}
    containers = payload.get("containers") if isinstance(payload.get("containers"), dict) else {}
    records = []
    for name in sorted(tools):
        container = containers.get(name, {}) if isinstance(containers.get(name), dict) else {}
        path_value = container.get("sif")
        expected = str(container.get("sha256", ""))
        path = Path(str(path_value)) if path_value else None
        actual = _sha256(path) if path and path.is_file() else None
        if not path_value or not expected:
            status = "missing-lock"
        elif actual is None:
            status = "missing-sif"
        elif actual != expected:
            status = "checksum-mismatch"
        else:
            status = "pass"
        records.append(
            {
                "tool": name,
                "version": tools[name].get("version") if isinstance(tools[name], dict) else None,
                "sif": path_value,
                "expected_sha256": expected or None,
                "actual_sha256": actual,
                "status": status,
            }
        )
    return {
        "status": "pass" if records and all(row["status"] == "pass" for row in records) else "fail",
        "lock": str(lock_path),
        "records": records,
        "policy": "production execution requires an immutable local SIF and matching SHA-256",
    }


def forecast_plan(plan_path: Path, output: Path | None = None) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tasks = plan.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("Plan tasks must be a list")
    pipeline_counts = Counter(str(task.get("pipeline")) for task in tasks)
    cpu_hours = 0.0
    gpu_hours = 0.0
    peak_memory_gb = 0.0
    estimated_storage_gb = 0.0
    input_bytes = 0
    for task in tasks:
        resources = task.get("resources", {})
        hours = _duration_hours(str(resources.get("time", "01:00:00")))
        cpus = max(1, int(resources.get("cpus", 1)))
        gpus = max(0, int(resources.get("gpus", 0)))
        cpu_hours += hours * cpus
        gpu_hours += hours * gpus
        peak_memory_gb = max(peak_memory_gb, float(resources.get("memory_gb", 0)))
        estimated_storage_gb += DEFAULT_STORAGE_GB.get(str(task.get("pipeline")), 0.1)
        for input_value in task.get("inputs", []):
            path = Path(str(input_value))
            if path.is_file():
                input_bytes += path.stat().st_size
    blocked = [str(task.get("task_id")) for task in tasks if not task.get("command")]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": str(plan_path),
        "tasks": len(tasks),
        "pipeline_counts": dict(sorted(pipeline_counts.items())),
        "blocked_command_count": len(blocked),
        "blocked_command_examples": blocked[:50],
        "resource_upper_bounds": {
            "cpu_core_hours": round(cpu_hours, 2),
            "gpu_hours": round(gpu_hours, 2),
            "peak_task_memory_gb": round(peak_memory_gb, 2),
            "estimated_derivative_storage_gb": round(estimated_storage_gb, 2),
            "known_input_storage_gb": round(input_bytes / 1024**3, 2),
        },
        "interpretation": (
            "CPU/GPU figures are requested-resource upper bounds, not measured runtime; "
            "calibrate defaults with modality pilots before cohort launch."
        ),
    }
    if output:
        _write_json(output, report)
    return report


def monitor_engine(database: Path, output: Path | None = None) -> dict[str, Any]:
    store = TaskDatabase(database)
    rows = store.task_rows()
    summary = store.summary()
    completed = [row for row in rows if row["status"] == "complete"]
    timestamps = [_parse_time(row.get("updated_at")) for row in completed]
    timestamps = [value for value in timestamps if value]
    created = [_parse_time(row.get("created_at")) for row in rows]
    created = [value for value in created if value]
    elapsed_hours = (
        (max(timestamps) - min(created)).total_seconds() / 3600
        if timestamps and created
        else 0.0
    )
    throughput = len(completed) / elapsed_hours if elapsed_hours > 0 else 0.0
    remaining = sum(row["status"] in {"pending", "running"} for row in rows)
    eta_hours = remaining / throughput if throughput > 0 else None
    failures = Counter(str(row["last_error"] or "unknown")[:240] for row in rows if row["status"] == "failed")
    report = {
        **summary,
        "completed_tasks": len(completed),
        "remaining_tasks": remaining,
        "throughput_tasks_per_hour": round(throughput, 3),
        "eta_hours_at_observed_throughput": round(eta_hours, 2) if eta_hours is not None else None,
        "failure_reasons": dict(failures.most_common(20)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if output:
        _write_json(output, report)
    return report


def write_exclusion_report(
    search_root: Path, output_tsv: Path, output_json: Path
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(search_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("status") not in {
            "fail",
            "partial",
            "review",
            "blocked",
        }:
            continue
        participant = payload.get("participant_id") or _entity(path.name, "sub")
        visit = payload.get("visit_id") or _entity(path.name, "ses")
        reasons = _exclusion_reasons(payload)
        if isinstance(reasons, str):
            reasons = [reasons]
        rows.append(
            {
                "participant_id": participant,
                "visit_id": visit,
                "status": payload.get("status"),
                "source_report": str(path),
                "reasons": ";".join(str(value) for value in reasons),
            }
        )
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_tsv.with_suffix(output_tsv.suffix + ".tmp")
    fields = ["participant_id", "visit_id", "status", "source_report", "reasons"]
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_tsv)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": len(rows),
        "status_counts": dict(Counter(str(row["status"]) for row in rows)),
        "tsv": str(output_tsv),
    }
    _write_json(output_json, {**summary, "exclusions": rows})
    return summary


def _exclusion_reasons(payload: dict[str, Any]) -> list[Any] | str:
    direct = (
        payload.get("failures")
        or payload.get("errors")
        or payload.get("warnings")
    )
    if direct:
        return direct
    nested: list[str] = []
    for record in payload.get("records", []):
        if not isinstance(record, dict) or record.get("status") != "fail":
            continue
        label = record.get("path") or record.get("modality") or "record"
        failures = record.get("failures") or ["failed-production-qc"]
        nested.extend(f"{label}:{reason}" for reason in failures)
    for record in payload.get("excluded_samples", []):
        if not isinstance(record, dict):
            continue
        label = record.get("output") or record.get("modality") or "sample"
        failures = record.get("failures") or [record.get("reason", "excluded")]
        nested.extend(f"{label}:{reason}" for reason in failures)
    return nested


def load_test_jsonl(
    input_path: Path, repetitions: int = 100, minimum_rows_per_second: float = 10000.0
) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("Load-test repetitions must be positive")
    lines = [line for line in input_path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        raise ValueError("Load-test input is empty")
    started = time.perf_counter()
    checksum = hashlib.sha256()
    parsed = 0
    for _ in range(repetitions):
        for line in lines:
            payload = json.loads(line)
            checksum.update(str(payload.get("participant_id", "")).encode("utf-8"))
            parsed += 1
    elapsed = time.perf_counter() - started
    throughput = parsed / max(elapsed, 1e-9)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return {
        "status": "pass" if throughput >= minimum_rows_per_second else "fail",
        "source_rows": len(lines),
        "simulated_rows": parsed,
        "elapsed_seconds": round(elapsed, 6),
        "rows_per_second": round(throughput, 2),
        "minimum_rows_per_second": minimum_rows_per_second,
        "peak_rss_mb": round(peak_mb, 2),
        "checksum": checksum.hexdigest(),
        "scope": "streaming manifest/index load; image-algorithm pilots are reported separately",
    }


def production_audit(
    plan_path: Path,
    container_lock: Path,
    validation_root: Path,
    output: Path,
    required_modalities: tuple[str, ...],
) -> dict[str, Any]:
    forecast = forecast_plan(plan_path)
    containers = verify_container_lock(container_lock)
    validated = _validated_modalities(validation_root)
    missing_pilots = sorted(set(required_modalities) - validated)
    failures = []
    if forecast["blocked_command_count"]:
        failures.append("plan-has-blocked-commands")
    if containers["status"] != "pass":
        failures.append("container-lock-incomplete")
    if missing_pilots:
        failures.append("modality-pilots-missing")
    report = {
        "status": "pass" if not failures else "fail",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "failures": failures,
        "forecast": forecast,
        "containers": containers,
        "validated_modalities": sorted(validated),
        "missing_validation_modalities": missing_pilots,
        "production_gate": (
            "No 50k-cohort launch until commands, immutable SIF checksums, and required "
            "real-data modality pilots all pass."
        ),
    }
    _write_json(output, report)
    return report


def _validated_modalities(root: Path) -> set[str]:
    modalities = set()
    if not root.exists():
        return modalities
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("status") == "pass" and payload.get("modality"):
            modalities.add(str(payload["modality"]))
    return modalities


def _duration_hours(value: str) -> float:
    if "-" in value:
        try:
            days, rest = value.split("-", 1)
            return int(days) * 24 + _duration_hours(rest)
        except ValueError:
            return 1.0
    parts = value.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (int(item) for item in parts)
            return hours + minutes / 60 + seconds / 3600
    except ValueError:
        pass
    return 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _entity(name: str, key: str) -> str | None:
    for token in name.split("_"):
        if token.startswith(key + "-"):
            return token.removeprefix(key + "-")
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
