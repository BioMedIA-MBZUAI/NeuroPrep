from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .experts import ExpertRegistry
from .inventory import inventory_dataset
from .models import DatasetConfig, Modality


ENTITY_PATTERN = re.compile(r"(?:^|_)([A-Za-z0-9]+)-([^_]+)")
DEFAULT_MODALITIES = (
    "t1w",
    "t2w",
    "flair",
    "pd",
    "fmri",
    "dwi",
    "pet",
    "asl",
    "swi",
    "mra",
    "ct",
    "eeg",
    "meg",
    "ieeg",
)


def assemble_visits(
    sample_index: Path,
    output_root: Path,
    expected_modalities: tuple[str, ...] = DEFAULT_MODALITIES,
    tolerance_days: int = 30,
    bids_root: Path | None = None,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """Assemble model-ready files into participant/visit manifests.

    BIDS sessions are authoritative. Records without a session are clustered by acquisition
    date, and only relative dates are written to avoid propagating date identifiers.
    """

    rows = _read_jsonl(sample_index)
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        participant = str(row.get("participant_id", "")).removeprefix("sub-")
        if not participant:
            raise ValueError("Every model-ready sample requires participant_id")
        if not _sample_passes(row):
            continue
        enriched = dict(row)
        enriched["_entities"] = _entities(Path(str(row.get("path", ""))))
        enriched["_time"] = _acquisition_time(enriched)
        by_subject[participant].append(enriched)

    expected_visits = _expected_visit_inventory(
        bids_root,
        dataset_id=dataset_id,
        output_root=output_root,
        tolerance_days=tolerance_days,
    )
    manifests: list[dict[str, Any]] = []
    participants = sorted(set(by_subject) | set(expected_visits))
    for participant in participants:
        subject_rows = by_subject.get(participant, [])
        released_groups = _visit_groups(subject_rows, tolerance_days)
        released_by_visit = {
            _public_visit_id(session, index): group
            for index, (session, group) in enumerate(released_groups, start=1)
        }
        expected_by_visit = {
            str(item["visit_id"]): item for item in expected_visits.get(participant, [])
        }
        ordered_visit_ids = [
            str(item["visit_id"]) for item in expected_visits.get(participant, [])
        ]
        ordered_visit_ids.extend(
            visit_id
            for visit_id in released_by_visit
            if visit_id not in expected_by_visit
        )
        if not ordered_visit_ids:
            continue
        dated = [
            _released_or_expected_time(
                released_by_visit.get(visit_id, []),
                expected_by_visit.get(visit_id),
            )
            for visit_id in ordered_visit_ids
        ]
        baseline = min((value for value in dated if value is not None), default=None)
        for visit_number, (visit_id, visit_time) in enumerate(
            zip(ordered_visit_ids, dated, strict=True), start=1
        ):
            visit_rows = released_by_visit.get(visit_id, [])
            expected_visit = expected_by_visit.get(visit_id, {})
            raw_modalities = sorted(
                {
                    str(value)
                    for value in expected_visit.get("raw_modalities", [])
                    if str(value) in expected_modalities
                }
            )
            anchor_sample = _reference_anchor(visit_rows)
            anchor = anchor_sample.get("_time") if anchor_sample else _visit_anchor(visit_rows)
            modalities: dict[str, dict[str, Any]] = {}
            for modality in sorted({str(row["modality"]) for row in visit_rows}):
                candidates = [row for row in visit_rows if row["modality"] == modality]
                ranked = sorted(
                    candidates,
                    key=lambda row: _selection_key(row, anchor, anchor_sample),
                )
                primary = _public_sample(ranked[0])
                modalities[modality] = {
                    "present": True,
                    "primary": primary,
                    "runs": [_public_sample(row) for row in ranked],
                    "alternates": [_public_sample(row) for row in ranked[1:]],
                    "selection_reason": (
                        "passing QC; scanner-compatible with anatomical anchor; declared common "
                        "standard space; closest acquisition time; highest spatial resolution; "
                        "deterministic path tie-break"
                    ),
                }
            missing = [name for name in expected_modalities if name not in modalities]
            for modality in missing:
                modalities[modality] = {
                    "present": False,
                    "primary": None,
                    "runs": [],
                    "alternates": [],
                    "selection_reason": "not acquired or not accepted by QC",
                }
            relative_days = (
                int(round((visit_time - baseline).total_seconds() / 86400))
                if visit_time is not None and baseline is not None
                else None
            )
            visit_key = _visit_key(
                participant,
                visit_id,
                visit_rows,
                [str(value) for value in expected_visit.get("raw_sources", [])],
            )
            site_covariates, site_conflicts = _site_covariates(visit_rows)
            available = sorted(set(modalities) - set(missing))
            processing_missing = sorted(set(raw_modalities) - set(available))
            manifest = {
                "contract_version": "3.0",
                "dataset_id": str(
                    visit_rows[0].get("dataset_id", "")
                    if visit_rows
                    else expected_visit.get("dataset_id", dataset_id or "")
                ),
                "participant_id": participant,
                "visit_id": visit_id,
                "visit_key": visit_key,
                "longitudinal_index": visit_number - 1,
                "days_from_baseline": relative_days,
                "acquisition_time_policy": (
                    "BIDS session or acquisition timestamps clustered within "
                    f"{tolerance_days} days; absolute dates are not exported"
                ),
                "anchor_sample_id": anchor_sample.get("sample_id") if anchor_sample else None,
                "compatibility_policy": (
                    "prefer the same scanner/field strength as the anatomical anchor and the "
                    "closest acquisition time; retain all alternates for audit"
                ),
                "modalities": modalities,
                "available_modalities": available,
                "missing_modalities": sorted(missing),
                "missing_modality_mask": {
                    modality: int(modality not in missing) for modality in expected_modalities
                },
                "raw_acquired_modalities": raw_modalities,
                "processing_missing_modalities": processing_missing,
                "processing_missing_modality_mask": {
                    modality: int(
                        modality in raw_modalities and modality not in available
                    )
                    for modality in expected_modalities
                },
                "site_covariates": site_covariates,
                "site_covariate_conflicts": site_conflicts,
                "sample_count": len(visit_rows),
                "status": "pass" if visit_rows else "excluded-no-released-modalities",
                "exclusion_reason": (
                    None
                    if visit_rows
                    else "all acquired modalities are missing, failed QC, or lack a valid "
                    "visit-local preprocessing anchor"
                ),
            }
            path = (
                output_root
                / f"sub-{participant}"
                / visit_id
                / f"sub-{participant}_{visit_id}_model-ready-visit.json"
            )
            _write_json(path, manifest)
            manifest["manifest_path"] = str(path)
            manifests.append(manifest)

    index_root = output_root / "index"
    index_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = index_root / "visits.jsonl"
    temporary = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for manifest in manifests:
            stream.write(json.dumps(manifest, sort_keys=True) + "\n")
    temporary.replace(jsonl_path)
    tsv_path = index_root / "visits.tsv"
    _write_visit_tsv(manifests, tsv_path, expected_modalities)
    summary = {
        "status": "pass" if manifests else "fail",
        "participants": len({manifest["participant_id"] for manifest in manifests}),
        "visits": len(manifests),
        "longitudinal_participants": sum(
            count > 1
            for count in _counts(manifest["participant_id"] for manifest in manifests).values()
        ),
        "excluded_visits": sum(
            manifest["status"] != "pass" for manifest in manifests
        ),
        "processing_missing_visits": sum(
            bool(manifest["processing_missing_modalities"]) for manifest in manifests
        ),
        "modality_presence": {
            modality: sum(modality in manifest["available_modalities"] for manifest in manifests)
            for modality in expected_modalities
        },
        "jsonl": str(jsonl_path),
        "tsv": str(tsv_path),
    }
    _write_json(index_root / "assembly_summary.json", summary)
    return summary


def _expected_visit_inventory(
    bids_root: Path | None,
    dataset_id: str | None,
    output_root: Path,
    tolerance_days: int,
) -> dict[str, list[dict[str, Any]]]:
    if bids_root is None:
        return {}
    root = Path(bids_root)
    if not root.exists():
        raise FileNotFoundError(f"BIDS root does not exist: {root}")
    identity = dataset_id or root.name
    inventory = inventory_dataset(
        DatasetConfig(
            dataset_id=identity,
            root=root,
            bids_root=root,
            derivatives_root=output_root,
            work_root=output_root / ".inventory-work",
        )
    )
    expected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for context in ExpertRegistry().visit_contexts(inventory.images, tolerance_days):
        raw_modalities = sorted(
            {
                image.modality.value
                for image in context.images
                if image.modality not in {Modality.UNKNOWN, Modality.FIELDMAP}
            }
        )
        expected[context.subject].append(
            {
                "dataset_id": identity,
                "visit_id": (
                    "visit-001" if context.visit_id == "single" else context.visit_id
                ),
                "visit_index": context.visit_index,
                "visit_count": context.visit_count,
                "acquisition_time": context.acquisition_time,
                "raw_modalities": raw_modalities,
                "raw_sources": sorted(str(image.path) for image in context.images),
            }
        )
    return expected


def _public_visit_id(session: str, index: int) -> str:
    return session if session != "single" else f"visit-{index:03d}"


def _released_or_expected_time(
    rows: list[dict[str, Any]], expected: dict[str, Any] | None
) -> datetime | None:
    released = min(
        (row["_time"] for row in rows if row["_time"] is not None),
        default=None,
    )
    if released is not None:
        return released
    if expected and expected.get("acquisition_time"):
        return _parse_datetime(str(expected["acquisition_time"]))
    return None


def _visit_groups(
    rows: list[dict[str, Any]], tolerance_days: int
) -> list[tuple[str, list[dict[str, Any]]]]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    undated: list[dict[str, Any]] = []
    for row in rows:
        session = row["_entities"].get("ses")
        if session:
            by_session[f"ses-{session.removeprefix('ses-')}"].append(row)
        else:
            undated.append(row)
    groups = sorted(by_session.items())
    dated = sorted(
        undated,
        key=lambda row: (
            row["_time"] or datetime.max.replace(tzinfo=timezone.utc),
            str(row["path"]),
        ),
    )
    timestamp_groups: list[list[dict[str, Any]]] = []
    for row in dated:
        if not timestamp_groups:
            timestamp_groups.append([row])
            continue
        previous_times = [item["_time"] for item in timestamp_groups[-1] if item["_time"]]
        anchor = min(previous_times) if previous_times else None
        if (
            row["_time"]
            and anchor
            and abs((row["_time"] - anchor).total_seconds()) <= max(0, tolerance_days) * 86_400
        ):
            timestamp_groups[-1].append(row)
        elif row["_time"] is None and anchor is None:
            timestamp_groups[-1].append(row)
        else:
            timestamp_groups.append([row])
    groups.extend(("single", group) for group in timestamp_groups)
    return sorted(
        groups,
        key=lambda item: (
            _visit_anchor(item[1]) or datetime.max.replace(tzinfo=timezone.utc),
            item[0],
        ),
    )


def _selection_key(
    row: dict[str, Any],
    anchor: datetime | None,
    anchor_sample: dict[str, Any] | None,
) -> tuple[Any, ...]:
    qc = row.get("qc") if isinstance(row.get("qc"), dict) else {}
    voxel_sizes = qc.get("voxel_sizes_mm") or qc.get("zooms") or [99, 99, 99]
    try:
        voxel_volume = float(voxel_sizes[0]) * float(voxel_sizes[1]) * float(voxel_sizes[2])
    except (IndexError, TypeError, ValueError):
        voxel_volume = 999999.0
    sample_time = row.get("_time")
    distance = abs((sample_time - anchor).total_seconds()) if sample_time and anchor else 0.0
    flags = qc.get("flags", []) if isinstance(qc, dict) else []
    incompatibilities = _scanner_incompatibilities(row, anchor_sample)
    return (
        len(flags),
        incompatibilities,
        _space_priority(row),
        distance,
        voxel_volume,
        str(row.get("path", "")),
    )


def _reference_anchor(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    priorities = {"t1w": 0, "t2w": 1, "flair": 2, "ct": 3}
    candidates = [row for row in rows if str(row.get("modality")) in priorities]
    if not candidates:
        candidates = list(rows)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            priorities.get(str(row.get("modality")), 99),
            len((row.get("qc") or {}).get("flags", [])),
            _space_priority(row),
            _voxel_volume(row),
            str(row.get("path", "")),
        ),
    )


def _scanner_incompatibilities(row: dict[str, Any], anchor: dict[str, Any] | None) -> int:
    if anchor is None:
        return 0
    fields = (
        "DeviceSerialNumber",
        "StationName",
        "Manufacturer",
        "ManufacturersModelName",
        "MagneticFieldStrength",
    )
    current = row.get("site_covariates")
    reference = anchor.get("site_covariates")
    if not isinstance(current, dict) or not isinstance(reference, dict):
        return 0
    return sum(
        str(current[field]) != str(reference[field])
        for field in fields
        if current.get(field) not in (None, "") and reference.get(field) not in (None, "")
    )


def _voxel_volume(row: dict[str, Any]) -> float:
    qc = row.get("qc") if isinstance(row.get("qc"), dict) else {}
    voxel_sizes = qc.get("voxel_sizes_mm") or qc.get("zooms") or [99, 99, 99]
    try:
        return float(voxel_sizes[0]) * float(voxel_sizes[1]) * float(voxel_sizes[2])
    except (IndexError, TypeError, ValueError):
        return 999999.0


def _space_priority(row: dict[str, Any]) -> int:
    """Prefer an already-normalized common space over a finer native derivative."""

    space = str(row.get("space", "")).lower()
    if "mni" in space:
        return 0
    if "acpc" in space:
        return 1
    if space in {"subject-anat", "subject", "anat"}:
        return 2
    if "native" in space:
        return 3
    return 4


def _sample_passes(row: dict[str, Any]) -> bool:
    qc = row.get("qc")
    return bool(row.get("path")) and isinstance(qc, dict) and qc.get("status") == "pass"


def _public_sample(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _visit_anchor(rows: list[dict[str, Any]]) -> datetime | None:
    values = sorted(row["_time"] for row in rows if row.get("_time") is not None)
    return values[len(values) // 2] if values else None


def _acquisition_time(row: dict[str, Any]) -> datetime | None:
    metadata_path = row.get("metadata")
    payload: dict[str, Any] = {}
    if metadata_path and Path(str(metadata_path)).exists():
        try:
            raw = json.loads(Path(str(metadata_path)).read_text(encoding="utf-8"))
            payload = raw if isinstance(raw, dict) else {}
        except (OSError, json.JSONDecodeError):
            pass
    for field in (
        "AcquisitionDateTime",
        "ScanDate",
        "StudyDateTime",
        "AcquisitionDate",
        "SeriesDate",
    ):
        if payload.get(field):
            if value := _parse_datetime(str(payload[field])):
                return value
    return None


def _parse_datetime(value: str) -> datetime | None:
    normalized = value.strip().replace("Z", "+00:00")
    formats = (
        None,
        "%Y%m%d%H%M%S.%f",
        "%Y%m%d%H%M%S",
        "%Y%m%d",
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
    )
    for format_string in formats:
        try:
            parsed = (
                datetime.fromisoformat(normalized)
                if format_string is None
                else datetime.strptime(normalized, format_string)
            )
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _entities(path: Path) -> dict[str, str]:
    entities = {
        match.group(1).lower(): match.group(2)
        for match in ENTITY_PATTERN.finditer(path.name)
    }
    if "ses" not in entities:
        session = next(
            (part.removeprefix("ses-") for part in reversed(path.parts) if part.startswith("ses-")),
            None,
        )
        if session:
            entities["ses"] = session
    return entities


def _visit_key(
    participant: str,
    visit_id: str,
    rows: list[dict[str, Any]],
    raw_sources: list[str] | None = None,
) -> str:
    sources = sorted(
        {
            *(str(row.get("sha256") or row.get("path")) for row in rows),
            *(raw_sources or []),
        }
    )
    digest = hashlib.sha256("\x1f".join(sources).encode("utf-8")).hexdigest()[:16]
    return f"sub-{participant}:{visit_id}:{digest}"


def _site_covariates(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        payload = row.get("site_covariates")
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if value not in (None, ""):
                values[str(key)].add(str(value))
    conflicts = sorted(key for key, observed in values.items() if len(observed) > 1)
    resolved = {key: next(iter(observed)) for key, observed in values.items() if len(observed) == 1}
    return resolved, conflicts


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _write_visit_tsv(
    manifests: list[dict[str, Any]], path: Path, expected_modalities: tuple[str, ...]
) -> None:
    fields = [
        "dataset_id",
        "participant_id",
        "visit_id",
        "visit_key",
        "longitudinal_index",
        "days_from_baseline",
        "sample_count",
        *[f"has_{modality}" for modality in expected_modalities],
        "manifest_path",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for manifest in manifests:
            row = {key: manifest.get(key) for key in fields}
            row.update(
                {
                    f"has_{modality}": int(modality in manifest["available_modalities"])
                    for modality in expected_modalities
                }
            )
            writer.writerow(row)
    temporary.replace(path)


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return dict(counts)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
