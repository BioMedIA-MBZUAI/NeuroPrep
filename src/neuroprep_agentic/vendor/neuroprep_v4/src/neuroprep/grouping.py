from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib

from .contracts import audit_image
from .models import DatasetConfig, ImageRecord, Inventory, Modality


SELECTION_POLICY_VERSION = "1.0"
STRUCTURAL_PRIORITY = {
    Modality.T1W: 4,
    Modality.T2W: 3,
    Modality.FLAIR: 2,
    Modality.CT: 1,
}
PRESERVE_ALL = {
    Modality.FMRI,
    Modality.DWI,
    Modality.PET,
    Modality.ASL,
    Modality.EEG,
    Modality.MEG,
    Modality.IEEG,
}
COMPONENT_MODALITIES = {Modality.FMRI, Modality.PET, Modality.SWI, Modality.MRA}


@dataclass
class GroupedRecord:
    subject: str
    session: str
    modality: str
    protocol_family: str
    protocol_group_id: str
    acquisition_group_id: str
    run: str
    component: str
    image_format: str
    metadata_status: str
    eligibility: str
    qc_status: str
    issue_codes: str
    voxel_size: str
    shape: str
    source_path: str
    selection_role: str = ""
    selection_score: float | None = None
    selection_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def group_inventory(
    inventory: Inventory,
    config: DatasetConfig,
    apply_selection: bool = False,
) -> list[GroupedRecord]:
    rows = [_group_image(image, config) for image in inventory.images if not image.is_derivative]
    if apply_selection:
        _select_roles(rows, config)
    rows.sort(
        key=lambda row: (
            row.subject,
            row.session,
            row.modality,
            row.protocol_family,
            row.acquisition_group_id,
            row.component,
            row.source_path,
        )
    )
    return rows


def manifest_summary(rows: Iterable[GroupedRecord], dataset_id: str) -> dict[str, Any]:
    records = list(rows)
    return {
        "dataset_id": dataset_id,
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "records": len(records),
        "subjects": len({row.subject for row in records}),
        "sessions": len({(row.subject, row.session) for row in records}),
        "unresolved_sessions": len(
            {(row.subject, row.session) for row in records if row.session == "unresolved"}
        ),
        "records_with_unresolved_session": sum(
            row.session == "unresolved" for row in records
        ),
        "protocol_groups": len({row.protocol_group_id for row in records}),
        "acquisition_groups": len({row.acquisition_group_id for row in records}),
        "eligibility_counts": dict(sorted(Counter(row.eligibility for row in records).items())),
        "selection_counts": dict(
            sorted(Counter(row.selection_role or "unassigned" for row in records).items())
        ),
        "modality_counts": dict(sorted(Counter(row.modality for row in records).items())),
    }


def select_primary_reference(
    images: Iterable[ImageRecord],
    config: DatasetConfig,
) -> ImageRecord | None:
    """Select the best eligible structural reference using production scoring."""

    candidates = list(images)
    rows = [_group_image(image, config) for image in candidates]
    _select_roles(rows, config)
    primary_paths = {
        row.source_path
        for row in rows
        if row.selection_role == "primary-reference"
    }
    return next(
        (
            image
            for image in candidates
            if str(image.path) in primary_paths
        ),
        None,
    )


def write_manifest(rows: list[GroupedRecord], path: Path, dataset_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix.lower() == ".json":
        payload = {
            "summary": manifest_summary(rows, dataset_id),
            "records": [row.to_dict() for row in rows],
        }
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    else:
        fieldnames = list(GroupedRecord.__dataclass_fields__)
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(row.to_dict() for row in rows)
    temporary.replace(path)


def _group_image(image: ImageRecord, config: DatasetConfig) -> GroupedRecord:
    metadata = _read_metadata(image.sidecar)
    geometry = _geometry(image)
    family = _protocol_family(image, metadata, geometry)
    protocol_signature = _protocol_signature(image, metadata, geometry, family)
    requires_visit = bool(
        config.metadata.get("requires_visit_mapping", config.adapter == "scan")
    )
    session = image.session or ("unresolved" if requires_visit else "single")
    protocol_group_id = _identifier(
        "pg",
        image.subject,
        session,
        image.modality.value,
        family,
        protocol_signature,
    )
    run = _run_label(image, metadata)
    acquisition_group_id = _identifier(
        "ag",
        image.subject,
        session,
        image.modality.value,
        family,
        run,
    )
    component = _component_label(image, metadata, geometry)
    issues = audit_image(image)
    issue_codes = sorted({issue.code for issue in issues})
    if requires_visit and image.session is None:
        issue_codes.append("scan_visit_unresolved")

    if image.subject == "__unknown__":
        eligibility = "manual-review"
        issue_codes.append("unknown_subject")
    elif image.modality is Modality.UNKNOWN:
        eligibility = "manual-review"
    elif image.format not in {"nifti", "electrophysiology"}:
        eligibility = "manual-review"
    elif geometry["header_status"] != "readable":
        eligibility = "manual-review"
        issue_codes.append("unreadable_header")
    elif (
        requires_visit and image.session is None
    ):
        eligibility = "manual-review"
    elif any(issue.severity == "error" for issue in issues):
        eligibility = "manual-review"
    else:
        eligibility = "eligible"

    metadata_status = "present" if image.sidecar else "missing"
    if image.format not in {"nifti", "electrophysiology"}:
        metadata_status = "pending-ingest"
    if image.format not in {"nifti", "electrophysiology"}:
        qc_status = "pending-ingest"
    elif geometry["header_status"] != "readable":
        qc_status = "header-fail"
    elif any(issue.severity == "error" for issue in issues):
        qc_status = "metadata-fail"
    else:
        qc_status = "not-run"

    return GroupedRecord(
        subject=image.subject,
        session=session,
        modality=image.modality.value,
        protocol_family=family,
        protocol_group_id=protocol_group_id,
        acquisition_group_id=acquisition_group_id,
        run=run,
        component=component,
        image_format=image.format,
        metadata_status=metadata_status,
        eligibility=eligibility,
        qc_status=qc_status,
        issue_codes=";".join(sorted(set(issue_codes))),
        voxel_size=geometry["voxel_size"],
        shape=geometry["shape"],
        source_path=str(image.path),
    )


def _select_roles(rows: list[GroupedRecord], config: DatasetConfig) -> None:
    for row in rows:
        if row.eligibility != "eligible":
            row.selection_role = "manual-review"
            row.selection_reason = row.issue_codes or "eligibility review required"
            continue
        modality = Modality(row.modality)
        if modality is Modality.FIELDMAP:
            row.selection_role = "calibration"
            row.selection_reason = "field map retained for linked distortion correction"
        elif row.component and modality in COMPONENT_MODALITIES:
            row.selection_role = "component"
            row.selection_reason = "component retained with its acquisition group"
        elif modality in PRESERVE_ALL or modality in {Modality.SWI, Modality.MRA, Modality.PD}:
            row.selection_role = "include"
            row.selection_reason = "all eligible runs are preserved for the modality pipeline"
        elif modality in STRUCTURAL_PRIORITY:
            row.selection_role = "alternate"
            row.selection_reason = "valid structural candidate pending reference ranking"
        else:
            row.selection_role = "manual-review"
            row.selection_reason = "no automatic selection policy is defined"

    by_visit: dict[tuple[str, str], list[GroupedRecord]] = {}
    for row in rows:
        by_visit.setdefault((row.subject, row.session), []).append(row)

    for visit_rows in by_visit.values():
        candidates = [
            row
            for row in visit_rows
            if row.eligibility == "eligible"
            and Modality(row.modality) in STRUCTURAL_PRIORITY
            and not row.component
        ]
        if not candidates:
            continue
        for candidate in candidates:
            candidate.selection_score = _reference_score(candidate, config)
        primary = max(
            candidates,
            key=lambda row: (
                row.selection_score or 0.0,
                _inverse_lexical(row.source_path),
            ),
        )
        primary.selection_role = "primary-reference"
        primary.selection_reason = "highest-ranked eligible structural reference for this visit"
        for candidate in candidates:
            if candidate is primary:
                continue
            if candidate.protocol_group_id == primary.protocol_group_id:
                candidate.selection_role = "include"
                candidate.selection_reason = "compatible repeat retained for structural averaging"


def _reference_score(row: GroupedRecord, config: DatasetConfig) -> float:
    modality = Modality(row.modality)
    score = float(STRUCTURAL_PRIORITY[modality] * 1000)
    preferences = config.metadata.get("protocol_preferences", {})
    preferred = preferences.get(row.modality, []) if isinstance(preferences, dict) else []
    if isinstance(preferred, list):
        for index, label in enumerate(preferred):
            if str(label).lower() in row.protocol_family.lower():
                score += 300 - index * 20
                break
    voxel_values = _parse_numbers(row.voxel_size)
    if len(voxel_values) >= 3 and all(value > 0 for value in voxel_values[:3]):
        spatial = voxel_values[:3]
        score += max(0.0, 100.0 - sum(spatial) / 3.0 * 20.0)
        score += max(0.0, 20.0 - (max(spatial) - min(spatial)) * 20.0)
    shape_values = _parse_numbers(row.shape)
    if len(shape_values) >= 3:
        score += min(shape_values[:3]) / 100.0
    if row.run in {"1", "01"}:
        score += 0.01
    return round(score, 4)


def _protocol_family(
    image: ImageRecord,
    metadata: dict[str, Any],
    geometry: dict[str, str],
) -> str:
    modality = image.modality.value
    entities = image.entities
    acquisition = _clean_label(entities.get("acq", ""))
    task = _clean_label(entities.get("task", ""))
    tracer = _clean_label(
        entities.get("trc", "")
        or str(metadata.get("TracerName") or metadata.get("Radiopharmaceutical") or "")
    )
    reconstruction = _clean_label(entities.get("rec", ""))
    text = " ".join(
        (
            image.path.name,
            str(metadata.get("SeriesDescription", "")),
            str(metadata.get("ProtocolName", "")),
            str(metadata.get("SequenceName", "")),
        )
    ).lower()

    labels: list[str] = [modality]
    if acquisition:
        labels.append(acquisition)
    elif image.modality is Modality.T1W:
        labels.append(_first_match(text, ("mprage", "bravo", "spgr", "tfl"), "t1"))
    elif image.modality is Modality.T2W:
        labels.append(_first_match(text, ("space", "cube", "tse", "propeller"), "t2"))
    elif image.modality is Modality.FLAIR:
        labels.append("3d" if "3d" in text else "flair")
    elif image.modality is Modality.FMRI:
        labels.append(task or ("rest" if "rest" in text else "bold"))
        multiband = metadata.get("MultibandAccelerationFactor")
        if multiband:
            labels.append(f"mb{_number_label(multiband)}")
        elif match := re.search(r"mb\s*([0-9]+)", text):
            labels.append(f"mb{match.group(1)}")
    elif image.modality is Modality.DWI:
        multipart = _clean_label(str(metadata.get("MultipartID", "")))
        labels.append(multipart or _dwi_shell_label(image.path))
    elif image.modality is Modality.PET:
        labels.extend(value for value in (tracer, reconstruction) if value)
    elif image.modality is Modality.ASL:
        labels.append(_first_match(text, ("pcasl", "pasl", "casl"), "asl"))
    elif image.modality is Modality.SWI:
        labels.append(_first_match(text, ("qsm", "swi", "t2star", "gre"), "swi"))
    elif image.modality is Modality.MRA:
        labels.append(_first_match(text, ("tof", "phasecontrast", "contrast"), "mra"))
    elif image.modality is Modality.CT:
        labels.append("contrast" if metadata.get("ContrastBolusAgent") else "noncontrast")
    elif image.modality in {Modality.EEG, Modality.MEG, Modality.IEEG}:
        labels.append(task or "recording")
        sampling_frequency = metadata.get("SamplingFrequency")
        if sampling_frequency:
            labels.append(f"hz{_number_label(sampling_frequency)}")

    if image.modality in {Modality.T1W, Modality.T2W, Modality.FLAIR, Modality.CT}:
        voxel = geometry.get("voxel_size")
        if voxel:
            labels.append(f"vox{voxel.replace('x', 'p')}")
    return "_".join(filter(None, labels))


def _protocol_signature(
    image: ImageRecord,
    metadata: dict[str, Any],
    geometry: dict[str, str],
    family: str,
) -> str:
    fields = (
        "MagneticFieldStrength",
        "Manufacturer",
        "ManufacturersModelName",
        "RepetitionTime",
        "FlipAngle",
        "PhaseEncodingDirection",
        "TotalReadoutTime",
        "MultibandAccelerationFactor",
        "ArterialSpinLabelingType",
        "M0Type",
        "Units",
        "MultipartID",
    )
    payload = {field: metadata.get(field) for field in fields if metadata.get(field) is not None}
    if "echo" not in image.entities:
        payload["EchoTime"] = metadata.get("EchoTime")
    payload.update(
        {
            "family": family,
            "voxel_size": geometry["voxel_size"],
            "spatial_shape": "x".join(geometry["shape"].split("x")[:3]),
        }
    )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _run_label(image: ImageRecord, metadata: dict[str, Any]) -> str:
    if run := image.entities.get("run"):
        return run
    for field in ("AcquisitionNumber", "TemporalPositionIdentifier"):
        if metadata.get(field) not in (None, ""):
            return str(metadata[field])
    if uid := metadata.get("AcquisitionUID"):
        return _identifier("run", str(uid))
    stem = image.path.name[:-7] if image.path.name.endswith(".nii.gz") else image.path.stem
    stem = re.sub(r"_(echo|part)-?[A-Za-z0-9]+", "", stem, flags=re.I)
    stem = re.sub(r"_i[0-9]+", "", stem, flags=re.I)
    if image.format == "archive" and image.entities.get("loniuid"):
        return image.entities["loniuid"]
    return _clean_label(stem) or _clean_label(image.path.parent.name) or "1"


def _component_label(
    image: ImageRecord,
    metadata: dict[str, Any],
    geometry: dict[str, str],
) -> str:
    components: list[str] = []
    for entity in ("echo", "part", "chunk", "rec"):
        if value := image.entities.get(entity):
            components.append(f"{entity}-{value}")
    name = image.path.name.lower()
    if match := re.search(r"_i([0-9]+)(?:_|\.)", name):
        components.append(f"image-{match.group(1)}")
    raw_image_type = metadata.get("ImageType", [])
    if isinstance(raw_image_type, list):
        image_type = " ".join(str(value) for value in raw_image_type)
    else:
        image_type = str(raw_image_type)
    for label, pattern in (
        ("magnitude", r"\b(mag|magnitude)\b"),
        ("phase", r"\b(p|phase)\b"),
        ("minip", r"minip"),
    ):
        if re.search(pattern, f"{name} {image_type}", re.I):
            components.append(label)
    shape = _parse_numbers(geometry["shape"])
    description = " ".join(
        (str(metadata.get("SeriesDescription", "")), str(metadata.get("ProtocolName", "")))
    )
    if image.modality is Modality.FMRI and (
        re.search(r"(?:^|[_ -])(ref|sbref)(?:$|[_ -])", description, re.I)
        or (len(shape) == 4 and shape[3] <= 5)
    ):
        components.append("reference")
    return ";".join(dict.fromkeys(components))


def _geometry(image: ImageRecord) -> dict[str, str]:
    if image.format == "electrophysiology":
        return {"voxel_size": "", "shape": "", "header_status": "not-applicable"}
    if image.format != "nifti":
        return {"voxel_size": "", "shape": "", "header_status": "pending-ingest"}
    try:
        loaded = nib.load(str(image.path))
        zooms = loaded.header.get_zooms()[:3]
        return {
            "voxel_size": "x".join(f"{float(value):.4g}" for value in zooms),
            "shape": "x".join(str(int(value)) for value in loaded.shape),
            "header_status": "readable",
        }
    except Exception:
        return {"voxel_size": "", "shape": "", "header_status": "unreadable"}


def _read_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _dwi_shell_label(path: Path) -> str:
    companion = path.with_suffix("") if path.name.endswith(".nii.gz") else path
    companion = companion.with_suffix(".bval")
    try:
        values = [round(float(value) / 50) * 50 for value in companion.read_text().split()]
    except (OSError, ValueError):
        return "dwi"
    shells = sorted({int(value) for value in values})
    return "b" + "-".join(str(value) for value in shells)


def _identifier(prefix: str, *values: str) -> str:
    digest = hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _clean_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())[:48]


def _first_match(text: str, labels: tuple[str, ...], fallback: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "", text)
    return next((label for label in labels if label in compact), fallback)


def _number_label(value: Any) -> str:
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number).replace(".", "p")
    except (TypeError, ValueError):
        return _clean_label(str(value))


def _parse_numbers(value: str) -> list[float]:
    try:
        return [float(item) for item in value.split("x") if item]
    except ValueError:
        return []


def _inverse_lexical(value: str) -> tuple[int, ...]:
    return tuple(-ord(character) for character in value)
