from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .execution_models import ExecutableTask, ExecutionPlan
from .settings import validated_cluster_path
from .state import RunState

TENSOR_SUFFIXES = {
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".pt",
    ".safetensors",
    ".zarr",
}


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _artifact_evidence(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "path": str(path),
            "artifact_type": "file",
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
    if path.is_dir():
        children = sum(1 for item in path.rglob("*") if item.is_file())
        return {
            "path": str(path),
            "artifact_type": "directory",
            "file_count": children,
            "mtime_ns": path.stat().st_mtime_ns,
        }
    raise FileNotFoundError(path)


def finalize_execution_run(
    plan: ExecutionPlan,
    state: RunState,
    output_root: str,
) -> dict[str, Any]:
    destination = validated_cluster_path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    records = state.task_records()
    planned = {task.task_id: task for task in plan.tasks}
    observed = {str(row["task_id"]) for row in records}
    if observed != set(planned):
        missing = sorted(set(planned) - observed)
        unexpected = sorted(observed - set(planned))
        raise ValueError(
            f"state/plan task mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    incomplete = [
        str(row["task_id"])
        for row in records
        if bool(row["required"]) and row["status"] != "complete"
    ]
    if incomplete:
        raise RuntimeError(f"{len(incomplete)} required tasks are not complete: {incomplete[:5]}")

    artifact_rows: list[dict[str, Any]] = []
    missing_artifacts: list[str] = []
    for row in records:
        if row["status"] != "complete":
            continue
        task = ExecutableTask.model_validate_json(row["task_json"])
        artifacts = json.loads(row["artifacts_json"] or "[]")
        for artifact in artifacts:
            path = Path(artifact)
            try:
                evidence = _artifact_evidence(path)
            except FileNotFoundError:
                missing_artifacts.append(str(path))
                continue
            if evidence.get("size") == 0 or evidence.get("file_count") == 0:
                missing_artifacts.append(str(path))
                continue
            artifact_rows.append(
                {
                    **evidence,
                    "dataset_id": plan.dataset_id,
                    "task_id": task.task_id,
                    "executor_id": task.executor_id,
                    "executor_version": task.metadata.get("executor_version"),
                    "task_fingerprint": task.fingerprint,
                    "subject_id": task.subject_id,
                    "session_id": task.session_id,
                    "acquisition_ids": task.acquisition_ids,
                    "plugin_ids": task.plugin_ids,
                    "stages": task.stages,
                }
            )
    if missing_artifacts:
        raise RuntimeError(
            f"{len(missing_artifacts)} recorded artifacts are missing or empty: "
            f"{missing_artifacts[:5]}"
        )

    model_ready_index = [
        {
            "record_type": "index_metadata",
            "dataset_id": plan.dataset_id,
            "artifact_count": len(artifact_rows),
            "generated_at": datetime.now(UTC).isoformat(),
        },
        *artifact_rows,
    ]

    tensor_rows = [
        row for row in artifact_rows if Path(str(row["path"])).suffix.lower() in TENSOR_SUFFIXES
    ]
    tensor_index = [
        {
            "record_type": "index_metadata",
            "dataset_id": plan.dataset_id,
            "tensor_count": len(tensor_rows),
            "generated_at": datetime.now(UTC).isoformat(),
        },
        *tensor_rows,
    ]
    status_counts = Counter(str(row["status"]) for row in records)
    summary = {
        "ready": True,
        "dataset_id": plan.dataset_id,
        "catalog_digest": plan.catalog_digest,
        "generated_at": datetime.now(UTC).isoformat(),
        "task_counts": dict(sorted(status_counts.items())),
        "artifact_count": len(artifact_rows),
        "tensor_count": len(tensor_rows),
        "artifact_index": str(destination / "model-ready-index.jsonl"),
        "tensor_index": str(destination / "tensor-index.jsonl"),
    }
    _atomic_jsonl(destination / "model-ready-index.jsonl", model_ready_index)
    _atomic_jsonl(destination / "tensor-index.jsonl", tensor_index)
    _atomic_json(destination / "release-summary.json", summary)
    return summary
