from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .engine import TaskDatabase
from .provenance import file_sha256


def build_release_records(database: Path, checksum_files: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in TaskDatabase(database).task_rows():
        spec = json.loads(row.pop("spec_json"))
        outputs = spec.get("outputs") or [None]
        decision = spec.get("params", {}).get("expert_decision") or {}
        for output_value in outputs:
            output = Path(output_value) if output_value else None
            exists = output.exists() if output else False
            record = {
                "contract_version": spec.get("params", {}).get(
                    "output_contract_version", "3.0"
                ),
                "dataset_id": row["dataset_id"],
                "participant_id": row["subject"],
                "session_id": row["session"],
                "task_id": row["task_id"],
                "task_fingerprint": row["fingerprint"],
                "stage": row["stage"],
                "pipeline": row["pipeline"],
                "expert": decision.get("expert") or spec.get("params", {}).get("expert"),
                "modalities": spec.get("params", {}).get("modalities", []),
                "payload_kind": decision.get("output_kind")
                or spec.get("params", {}).get("output_kind"),
                "native_space": decision.get("native_space"),
                "inputs": spec.get("inputs", []),
                "output": str(output) if output else None,
                "output_exists": exists,
                "output_type": (
                    "directory" if exists and output and output.is_dir() else "file" if exists else None
                ),
                "status": row["status"],
                "attempts": row["attempts"],
                "return_code": row["return_code"],
                "updated_at": row["updated_at"],
                "log_path": row["log_path"],
                "last_error": row["last_error"],
                "checksum_sha256": (
                    file_sha256(output)
                    if checksum_files and exists and output and output.is_file()
                    else None
                ),
            }
            records.append(record)
    return records


def write_release_index(
    database: Path,
    output: Path,
    checksum_files: bool = False,
) -> dict[str, Any]:
    records = build_release_records(database, checksum_files=checksum_files)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(output)
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database": str(database),
        "index": str(output),
        "records": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "release_ready": bool(records)
        and all(
            record["status"] == "complete"
            and (record["output"] is None or record["output_exists"])
            for record in records
        ),
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
