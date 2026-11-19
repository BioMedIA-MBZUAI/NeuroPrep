from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_subject_model_ready_manifest(
    dataset_id: str,
    subject: str,
    finalization_markers: list[Path],
    qc_report: Path,
    output: Path,
) -> dict[str, Any]:
    """Create a fail-closed subject manifest consumed directly by model loaders."""

    subject_token = subject.removeprefix("sub-")
    failures: list[str] = []
    records: list[dict[str, Any]] = []
    upstream_exclusions: list[dict[str, Any]] = []
    for marker in finalization_markers:
        payload = _read_object(marker, "finalization marker")
        status = str(payload.get("status", "fail")).lower()
        modality = str(payload.get("modality", "t1w" if "anat" in marker.name else "unknown"))
        if status != "pass":
            failures.append(f"failed-finalizer:{modality}:{marker}")
        marker_records = payload.get("records")
        if not isinstance(marker_records, list) or not marker_records:
            exclusions = payload.get("excluded_records")
            if status == "pass" and isinstance(exclusions, list) and exclusions:
                upstream_exclusions.extend(
                    {
                        "modality": modality,
                        "output": record.get("output"),
                        "reason": record.get("reason", "upstream-acquisition-excluded"),
                        "failures": [],
                        "metrics": {
                            "payload_kind": record.get("payload_kind"),
                            "shape": record.get("shape"),
                            "excluded_from": record.get("excluded_from", []),
                        },
                    }
                    for record in exclusions
                    if isinstance(record, dict)
                )
            else:
                failures.append(f"empty-finalizer:{modality}:{marker}")
            continue
        for record in marker_records:
            if not isinstance(record, dict):
                failures.append(f"invalid-record:{modality}:{marker}")
                continue
            output_value = record.get("output")
            output_path = Path(str(output_value)) if output_value else None
            if output_path is None or not output_path.is_file():
                failures.append(f"missing-payload:{modality}:{output_value}")
            qc = record.get("qc")
            if not isinstance(qc, dict) or qc.get("status") != "pass":
                failures.append(f"failed-payload-qc:{modality}:{output_value}")
            records.append(record)

    qc = _read_object(qc_report, "subject QC report")
    qc_status = str(qc.get("status", "fail")).lower()
    if qc_status not in {"pass", "partial"}:
        failures.append(f"failed-subject-qc:{qc_report}")
    failed_qc: dict[str, dict[str, Any]] = {
        str(record.get("path")): record
        for record in qc.get("records", [])
        if isinstance(record, dict) and record.get("status") == "fail"
    }
    excluded_samples: list[dict[str, Any]] = list(upstream_exclusions)
    released_records: list[dict[str, Any]] = []
    for record in records:
        output_value = str(record.get("output", ""))
        decision = failed_qc.get(output_value)
        if decision is None:
            released_records.append(record)
            continue
        excluded_samples.append(
            {
                "modality": str(record.get("modality", "unknown")),
                "output": output_value,
                "reason": "failed-production-qc",
                "failures": list(decision.get("failures", [])),
                "metrics": dict(decision.get("metrics", {})),
            }
        )
    if qc_status == "partial" and not excluded_samples:
        failures.append(f"partial-qc-without-excluded-records:{qc_report}")
    records = released_records
    seen_modalities = {str(record.get("modality", "unknown")) for record in records}

    status = "pass" if records and not failures else "fail"
    manifest = {
        "contract_version": "3.0",
        "dataset_id": dataset_id,
        "participant_id": subject_token,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "release_tier": (
            "model-ready-partial"
            if status == "pass" and excluded_samples
            else "model-ready"
            if status == "pass"
            else "blocked"
        ),
        "modalities": sorted(seen_modalities),
        "sample_count": len(records),
        "samples": records,
        "excluded_sample_count": len(excluded_samples),
        "excluded_samples": excluded_samples,
        "qc_report": str(qc_report),
        "failures": sorted(set(failures)),
        "loader_contract": {
            "index": "this JSON document",
            "image_payloads": "NIfTI paths in samples[].output; masks are already applied",
            "sensor_payloads": "clean FIF paths in samples[].output",
            "auxiliaries": "gradient/confound/channel/event paths in samples[].auxiliaries",
            "normalization": "Spatial and signal preparation are recorded per sample; no implicit transform is allowed",
        },
    }
    _write_json(output, manifest)
    return manifest


def index_model_ready_subjects(root: Path, output_jsonl: Path, output_tsv: Path) -> dict[str, Any]:
    """Aggregate passing subject manifests into deterministic JSONL and flat TSV indexes."""

    layout_consolidation = consolidate_legacy_anatomical_release(root)
    manifests = sorted(root.rglob("sub-*_model-ready.json"))
    rows: list[dict[str, Any]] = []
    blocked = 0
    for path in manifests:
        manifest = _read_object(path, "model-ready subject manifest")
        if manifest.get("status") != "pass":
            blocked += 1
            continue
        for index, record in enumerate(manifest.get("samples", [])):
            if not isinstance(record, dict):
                continue
            row = {
                "dataset_id": manifest.get("dataset_id"),
                "participant_id": manifest.get("participant_id"),
                "sample_id": f"{manifest.get('participant_id')}:{index:04d}",
                "modality": record.get("modality"),
                "payload_kind": record.get("payload_kind"),
                "space": record.get("space"),
                "units": record.get("units"),
                "site_covariates": record.get("site_covariates", {}),
                "path": record.get("output"),
                "metadata": record.get("metadata"),
                "mask": record.get("mask"),
                "auxiliaries": record.get("auxiliaries", {}),
                "visual_reports": record.get("visual_reports", []),
                "sha256": record.get("sha256"),
                "qc": record.get("qc"),
            }
            rows.append(row)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary_jsonl = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
    with temporary_jsonl.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary_jsonl.replace(output_jsonl)

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    temporary_tsv = output_tsv.with_suffix(output_tsv.suffix + ".tmp")
    columns = (
        "dataset_id",
        "participant_id",
        "sample_id",
        "modality",
        "payload_kind",
        "space",
        "units",
        "site_covariates_json",
        "path",
        "metadata",
        "mask",
        "auxiliaries_json",
        "visual_reports_json",
        "sha256",
    )
    with temporary_tsv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(
                            row["auxiliaries"]
                            if key == "auxiliaries_json"
                            else row["site_covariates"]
                            if key == "site_covariates_json"
                            else row["visual_reports"],
                            sort_keys=True,
                        )
                        if key
                        in {
                            "auxiliaries_json",
                            "site_covariates_json",
                            "visual_reports_json",
                        }
                        else row.get(key)
                    )
                    for key in columns
                }
            )
    temporary_tsv.replace(output_tsv)
    summary = {
        "status": "pass" if rows and not blocked else "fail",
        "subjects": len(manifests),
        "blocked_subjects": blocked,
        "samples": len(rows),
        "jsonl": str(output_jsonl),
        "tsv": str(output_tsv),
        "layout_consolidation": layout_consolidation,
    }
    _write_json(output_jsonl.with_suffix(output_jsonl.suffix + ".summary.json"), summary)
    return summary


def consolidate_legacy_anatomical_release(root: Path) -> dict[str, Any]:
    """Merge the former anatomy release tree into ``neuroprep-model-ready``.

    Dataset indexing is the safe migration point: every subject release task
    has completed and training assembly has not started. Files move atomically
    on the derivatives filesystem, collisions are allowed only for identical
    content, and stored provenance paths are updated before the index is built.
    """

    # Preserve the caller's lexical path so provenance strings using a symlinked
    # mount (for example /var versus /private/var) can be rewritten exactly.
    root = root.expanduser().absolute()
    legacy_root = root.parent / "neuroprep-anat"
    if not legacy_root.exists():
        return {
            "status": "not-needed",
            "legacy_root": str(legacy_root),
            "model_ready_root": str(root),
            "moved_files": 0,
            "deduplicated_files": 0,
            "rewritten_files": 0,
        }
    if not legacy_root.is_dir():
        raise ValueError(f"Legacy anatomical release is not a directory: {legacy_root}")

    moved_files = 0
    deduplicated_files = 0
    replacements: dict[str, str] = {str(legacy_root): str(root)}
    sources = sorted(path for path in legacy_root.rglob("*") if path.is_file())
    for source in sources:
        relative = source.relative_to(legacy_root)
        destination = root / relative
        if relative.name == "finalization.json" and len(relative.parts) == 2:
            destination = destination.with_name("t1w-finalization.json")
            replacements[str(source)] = str(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not destination.is_file() or not _same_file_content(source, destination):
                raise FileExistsError(
                    f"Refusing to overwrite a different consolidated release file: {destination}"
                )
            source.unlink()
            deduplicated_files += 1
            continue
        os.replace(source, destination)
        moved_files += 1

    rewritten_files = 0
    for rewrite_root in (root, root.parent / "neuroprep-qc"):
        if not rewrite_root.exists():
            continue
        candidates = sorted(path for path in rewrite_root.rglob("*") if path.is_file())
        for path in candidates:
            if path.suffix.lower() not in {".json", ".jsonl", ".tsv", ".html"}:
                continue
            try:
                original = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            updated = original
            for old, new in sorted(
                replacements.items(), key=lambda item: len(item[0]), reverse=True
            ):
                updated = updated.replace(old, new)
            if updated != original:
                _atomic_write_text(path, updated)
                rewritten_files += 1

    for directory in sorted(
        (path for path in legacy_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        legacy_root.rmdir()
    except OSError as exc:
        remaining = sorted(str(path) for path in legacy_root.rglob("*"))
        raise RuntimeError(
            f"Legacy anatomical release was not fully consolidated: {remaining[:20]}"
        ) from exc

    return {
        "status": "consolidated",
        "legacy_root": str(legacy_root),
        "model_ready_root": str(root),
        "moved_files": moved_files,
        "deduplicated_files": deduplicated_files,
        "rewritten_files": rewritten_files,
    }


def _same_file_content(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    return _sha256(first) == _sha256(second)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".layout.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label.capitalize()} must contain a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
