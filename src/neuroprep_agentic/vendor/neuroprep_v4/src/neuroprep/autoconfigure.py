from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml

from .classification import classify_suffix, infer_suffix
from .models import DatasetConfig, ImageRecord, Inventory, Modality
from .profiles import execution_profile


POLICY_VERSION = "3.0"


@dataclass(frozen=True)
class RecordInference:
    source_path: str
    subject: str
    session: str
    modality: str
    confidence: float
    evidence: tuple[str, ...]
    protocol_fingerprint: str
    protocol_family: str
    status: str
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = list(self.evidence)
        payload["blockers"] = list(self.blockers)
        payload["warnings"] = list(self.warnings)
        return payload


def autoconfigure_dataset(
    inventory: Inventory,
    config: DatasetConfig,
    output_dir: Path,
    minimum_confidence: float = 0.90,
) -> dict[str, Any]:
    """Infer reusable acquisition policies and isolate genuinely ambiguous protocols.

    Fingerprints intentionally exclude participant and visit identifiers. One reviewed
    decision therefore covers every acquisition with the same scanner/protocol signature.
    """

    records = [
        infer_record(image, minimum_confidence)
        for image in inventory.images
        if not image.is_derivative
    ]
    qsm_ready_paths = _qsm_ready_paths(inventory.images)
    records = [
        replace(
            record,
            warnings=tuple(
                warning
                for warning in record.warnings
                if warning != "qsm-component-requires-cluster-validation"
            ),
            status=(
                "blocked"
                if record.blockers
                else "ready"
                if record.confidence >= minimum_confidence
                else "review"
            ),
            evidence=tuple((*record.evidence, "matched-qsm-magnitude-phase-components")),
        )
        if record.source_path in qsm_ready_paths
        else record
        for record in records
    ]
    clusters: dict[str, list[RecordInference]] = defaultdict(list)
    for record in records:
        clusters[record.protocol_fingerprint].append(record)

    cluster_rows: list[dict[str, Any]] = []
    approved_modalities: set[str] = set()
    reviewed_modalities: set[str] = set()
    for fingerprint, members in sorted(clusters.items()):
        statuses = Counter(member.status for member in members)
        status = "blocked" if statuses["blocked"] else "review" if statuses["review"] else "ready"
        modality = members[0].modality
        blockers = sorted({item for member in members for item in member.blockers})
        warnings = sorted({item for member in members for item in member.warnings})
        row = {
            "protocol_fingerprint": fingerprint,
            "modality": modality,
            "protocol_family": members[0].protocol_family,
            "status": status,
            "confidence_min": min(member.confidence for member in members),
            "confidence_mean": round(
                sum(member.confidence for member in members) / len(members), 4
            ),
            "records": len(members),
            "subjects": len({member.subject for member in members}),
            "sessions": len({(member.subject, member.session) for member in members}),
            "blockers": ";".join(blockers),
            "warnings": ";".join(warnings),
            "example": members[0].source_path,
        }
        cluster_rows.append(row)
        if status == "ready":
            approved_modalities.add(modality)
        else:
            reviewed_modalities.add(modality)

    approved_modalities -= reviewed_modalities
    profile = execution_profile(config)
    generated_policy = {
        "version": POLICY_VERSION,
        "dataset_id": inventory.dataset_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "minimum_confidence": minimum_confidence,
        "approval_rule": (
            "A modality is auto-approved only when every observed protocol cluster is ready; "
            "review and blocked clusters remain fail-closed."
        ),
        "auto_approved_modalities": sorted(approved_modalities),
        "needs_review_modalities": sorted(reviewed_modalities),
        "cluster_policies": {
            row["protocol_fingerprint"]: {
                "modality": row["modality"],
                "protocol_family": row["protocol_family"],
                "status": row["status"],
                "executor": executor_for_modality(row["modality"], profile.name),
                "confidence": row["confidence_min"],
                "blockers": row["blockers"].split(";") if row["blockers"] else [],
                "warnings": row["warnings"].split(";") if row["warnings"] else [],
            }
            for row in cluster_rows
        },
        "defaults": _default_policies(config),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "dataset_inventory.tsv"
    inferred_path = output_dir / "inferred_modalities.tsv"
    clusters_path = output_dir / "protocol_clusters.tsv"
    review_path = output_dir / "needs_review.tsv"
    policy_path = output_dir / "generated_policy.yaml"
    _write_tsv([record.to_dict() for record in records], inferred_path)
    _write_tsv(cluster_rows, clusters_path)
    _write_tsv(
        [row for row in cluster_rows if row["status"] != "ready"],
        review_path,
        fieldnames=list(cluster_rows[0]) if cluster_rows else None,
    )
    _write_inventory(inventory, inventory_path)
    temporary_policy = policy_path.with_suffix(policy_path.suffix + ".tmp")
    temporary_policy.write_text(
        yaml.safe_dump(generated_policy, sort_keys=False), encoding="utf-8"
    )
    temporary_policy.replace(policy_path)
    summary = {
        "status": "review" if reviewed_modalities else "pass",
        "records": len(records),
        "protocol_clusters": len(cluster_rows),
        "auto_approved_modalities": sorted(approved_modalities),
        "needs_review_clusters": sum(row["status"] != "ready" for row in cluster_rows),
        "outputs": {
            "dataset_inventory": str(inventory_path),
            "inferred_modalities": str(inferred_path),
            "protocol_clusters": str(clusters_path),
            "generated_policy": str(policy_path),
            "needs_review": str(review_path),
        },
    }
    _write_json(output_dir / "autoconfigure_summary.json", summary)
    return summary


def infer_record(image: ImageRecord, minimum_confidence: float = 0.90) -> RecordInference:
    metadata = _read_json(image.sidecar)
    evidence: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []

    suffix_modality = classify_suffix(infer_suffix(image.path))
    if suffix_modality is image.modality and image.modality is not Modality.UNKNOWN:
        confidence = 0.99
        evidence.append("bids-suffix")
    elif str(metadata.get("Modality", "")).upper() in {"CT", "PT", "PET", "MR", "EEG", "MEG"}:
        confidence = 0.95
        evidence.append("acquisition-modality-tag")
    elif image.modality is not Modality.UNKNOWN:
        confidence = 0.80
        evidence.append("description-rule")
    else:
        confidence = 0.0
        blockers.append("unknown-modality")

    if image.sidecar:
        evidence.append("sidecar-metadata")
    else:
        warnings.append("sidecar-missing")
    _validate_prerequisites(image, metadata, blockers, warnings)

    if blockers:
        status = "blocked"
    elif confidence < minimum_confidence or warnings:
        status = "review"
    else:
        status = "ready"
    family, fingerprint = _protocol_identity(image, metadata)
    return RecordInference(
        source_path=str(image.path),
        subject=image.subject,
        session=image.session or "single",
        modality=image.modality.value,
        confidence=confidence,
        evidence=tuple(evidence),
        protocol_fingerprint=fingerprint,
        protocol_family=family,
        status=status,
        blockers=tuple(sorted(set(blockers))),
        warnings=tuple(sorted(set(warnings))),
    )


def load_generated_policy(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or str(payload.get("version")) != POLICY_VERSION:
        raise ValueError(f"Not a NeuroPrep {POLICY_VERSION} generated policy: {path}")
    return payload


def executor_for_modality(modality: str, profile: str = "high-throughput") -> str:
    structural = "deepprep" if profile == "high-throughput" else "smriprep"
    functional = (
        "deepprep + resting-state postprocess when applicable"
        if profile == "high-throughput"
        else "fmriprep+xcpd"
    )
    return {
        "t1w": structural,
        "t2w": "neuroprep-structural",
        "flair": "neuroprep-structural",
        "pd": "neuroprep-structural",
        "fmri": functional,
        "dwi": "qsiprep",
        "pet": "petprep",
        "asl": "aslprep",
        "swi": "neuroprep-susceptibility (QSMxT or reconstructed-SWI branch)",
        "mra": "neuroprep-mra",
        "ct": "neuroprep-ct",
        "eeg": "mne-bids-pipeline-eeg",
        "meg": "mne-bids-pipeline-meg",
        "ieeg": "neuroprep-sensor-ieeg",
    }.get(modality, "blocked")


def _validate_prerequisites(
    image: ImageRecord,
    metadata: dict[str, Any],
    blockers: list[str],
    warnings: list[str],
) -> None:
    modality = image.modality
    if modality is Modality.DWI:
        for extension in (".bval", ".bvec"):
            if not _replace_nifti_extension(image.path, extension).exists():
                blockers.append(f"dwi-{extension[1:]}-missing")
        if not metadata.get("PhaseEncodingDirection"):
            warnings.append("dwi-phase-encoding-missing")
    elif modality is Modality.CT:
        units = str(metadata.get("Units", "")).upper()
        has_scaling = metadata.get("RescaleSlope") is not None
        if units != "HU" and not has_scaling and not _plausible_ct_hu(image.path):
            blockers.append("ct-hu-scaling-unverified")
    elif modality is Modality.PET:
        if not (metadata.get("TracerName") or metadata.get("Radiopharmaceutical")):
            blockers.append("pet-tracer-missing")
        if not metadata.get("Units"):
            blockers.append("pet-units-missing")
        if len(_shape(image.path)) == 4 and not (
            metadata.get("FrameTimesStart") and metadata.get("FrameDuration")
        ):
            blockers.append("pet-dynamic-timing-missing")
    elif modality is Modality.ASL:
        if not metadata.get("ArterialSpinLabelingType"):
            warnings.append("asl-labeling-type-missing")
    elif modality is Modality.SWI:
        name = image.path.name.lower()
        image_type = " ".join(str(value) for value in metadata.get("ImageType", []))
        if not any(token in f"{name} {image_type.lower()}" for token in ("phase", "part-phase")):
            warnings.append("qsm-component-requires-cluster-validation")
        if metadata.get("EchoTime") is None:
            blockers.append("qsm-echo-time-missing")
    elif modality is Modality.MRA:
        text = " ".join(
            str(metadata.get(field, ""))
            for field in ("SeriesDescription", "ProtocolName", "SequenceName")
        ).lower()
        if not any(token in text for token in ("tof", "angio", "mra", "phase contrast")):
            warnings.append("mra-subtype-uncertain")
    elif modality in {Modality.EEG, Modality.MEG, Modality.IEEG}:
        if not _companion(image.path, "channels.tsv").exists():
            blockers.append("sensor-channels-missing")
        task = image.entities.get("task") or metadata.get("TaskName")
        if task and str(task).lower() not in {"rest", "resting", "restingstate"}:
            if not _companion(image.path, "events.tsv").exists():
                warnings.append("sensor-event-semantics-review")
        if modality is Modality.IEEG:
            for suffix in ("electrodes.tsv", "coordsystem.json"):
                if not _companion(image.path, suffix).exists():
                    blockers.append(f"ieeg-{suffix.replace('.', '-')}-missing")


def _protocol_identity(image: ImageRecord, metadata: dict[str, Any]) -> tuple[str, str]:
    geometry: dict[str, Any] = {}
    if image.format == "nifti":
        try:
            loaded = nib.load(str(image.path))
            geometry = {
                "shape": list(loaded.shape),
                "zooms": [round(float(value), 4) for value in loaded.header.get_zooms()],
            }
        except Exception:
            geometry = {"shape": "unreadable"}
    family_parts = [image.modality.value]
    for value in (
        image.entities.get("acq"),
        image.entities.get("task"),
        image.entities.get("trc"),
        metadata.get("SeriesDescription"),
        metadata.get("ProtocolName"),
    ):
        if value:
            family_parts.append(_clean(str(value)))
    family = "_".join(dict.fromkeys(part for part in family_parts if part))[:160]
    fields = (
        "Manufacturer",
        "ManufacturersModelName",
        "MagneticFieldStrength",
        "SequenceName",
        "PulseSequenceName",
        "RepetitionTime",
        "EchoTime",
        "FlipAngle",
        "PhaseEncodingDirection",
        "TotalReadoutTime",
        "MultibandAccelerationFactor",
        "ArterialSpinLabelingType",
        "M0Type",
        "Units",
        "TracerName",
        "Radiopharmaceutical",
        "BodyPartExamined",
        "ConvolutionKernel",
    )
    payload = {
        "modality": image.modality.value,
        "family": family,
        "metadata": {field: metadata.get(field) for field in fields if metadata.get(field) is not None},
        "geometry": geometry,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return family, f"protocol-{digest}"


def _qsm_ready_paths(images: list[ImageRecord]) -> set[str]:
    groups: dict[tuple[str, str, str], list[tuple[ImageRecord, str, Any]]] = defaultdict(list)
    for image in images:
        if image.modality is not Modality.SWI:
            continue
        metadata = _read_json(image.sidecar)
        text = " ".join(
            (
                image.path.name,
                str(metadata.get("ImageType", "")),
                str(image.entities.get("part", "")),
            )
        ).lower()
        component = "phase" if "phase" in text or "part-phase" in text else "magnitude"
        acquisition = image.entities.get("acq") or str(metadata.get("ProtocolName", "default"))
        groups[(image.subject, image.session or "single", acquisition)].append(
            (image, component, metadata.get("EchoTime"))
        )
    ready: set[str] = set()
    for members in groups.values():
        components = {component for _, component, _ in members}
        echo_times = {str(echo) for _, _, echo in members if echo is not None}
        if components == {"magnitude", "phase"} and echo_times:
            ready.update(str(image.path) for image, _, _ in members)
    return ready


def _default_policies(config: DatasetConfig) -> dict[str, Any]:
    return {
        "visit_tolerance_days": int(config.metadata.get("visit_tolerance_days", 30)),
        "common_grid": {
            "space": str(config.metadata.get("model_ready_space", "MNI152NLin2009cAsym")),
            "resolution_mm": float(config.metadata.get("model_ready_resolution_mm", 2)),
            "orientation": "RAS+",
            "single_resampling": True,
        },
        "selection": {
            "preserve_all_raw_runs": True,
            "one_primary_structural_reference_per_visit": True,
            "resolve_ties": "quality-then-time-then-deterministic-path",
        },
        "uncertain_protocol_action": "quarantine",
    }


def _write_inventory(inventory: Inventory, path: Path) -> None:
    rows = [
        {
            "dataset_id": inventory.dataset_id,
            "participant_id": image.subject,
            "session_id": image.session or "single",
            "modality": image.modality.value,
            "format": image.format,
            "path": str(image.path),
            "sidecar": str(image.sidecar) if image.sidecar else "",
        }
        for image in inventory.images
        if not image.is_derivative
    ]
    _write_tsv(rows, path)


def _write_tsv(
    rows: list[dict[str, Any]], path: Path, fieldnames: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or (list(rows[0]) if rows else ["status"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _shape(path: Path) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in nib.load(str(path)).shape)
    except Exception:
        return ()


def _plausible_ct_hu(path: Path) -> bool:
    try:
        image = nib.load(str(path))
        slices = tuple(slice(None, None, max(1, size // 64)) for size in image.shape[:3])
        values = np.asarray(image.dataobj[slices], dtype=np.float32)
        finite = values[np.isfinite(values)]
    except Exception:
        return False
    if finite.size < 100:
        return False
    low, high = np.percentile(finite, (1, 99))
    return -1200 <= float(low) <= 200 and 20 <= float(high) <= 4000


def _replace_nifti_extension(path: Path, extension: str) -> Path:
    stem = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    return path.with_name(stem + extension)


def _companion(path: Path, suffix: str) -> Path:
    stem = path.name
    for ending in (".nii.gz", ".fif", ".edf", ".bdf", ".vhdr", ".set", ".nwb"):
        if stem.endswith(ending):
            stem = stem[: -len(ending)]
            break
    if "_" in stem:
        stem = stem.rsplit("_", 1)[0]
    return path.with_name(f"{stem}_{suffix}")


def _clean(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())[:64]
