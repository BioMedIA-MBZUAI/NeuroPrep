from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib

from .models import ImageRecord, Inventory, Modality


@dataclass(frozen=True)
class ContractIssue:
    severity: str
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def audit_inventory(inventory: Inventory, max_examples: int = 50) -> dict[str, Any]:
    """Audit acquisition metadata needed by modality-specific pipelines.

    The returned examples are bounded so an archive-heavy dataset cannot create
    a multi-gigabyte error report. Counts always describe the complete inventory.
    """

    issues: list[ContractIssue] = []
    counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    for image in inventory.images:
        for issue in audit_image(image):
            counts[issue.code] += 1
            severity_counts[issue.severity] += 1
            if len(issues) < max_examples:
                issues.append(issue)
    return {
        "dataset_id": inventory.dataset_id,
        "records_checked": len(inventory.images),
        "issue_counts": dict(sorted(counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "examples_truncated": sum(counts.values()) > len(issues),
        "examples": [issue.to_dict() for issue in issues],
    }


def audit_image(image: ImageRecord) -> list[ContractIssue]:
    if image.format == "electrophysiology" and image.modality in {
        Modality.EEG,
        Modality.MEG,
        Modality.IEEG,
    }:
        return _audit_electrophysiology(image)
    if image.format != "nifti":
        return [
            _issue(
                "error",
                "ingest_required",
                image,
                "Archive/DICOM input must be curated to BIDS before preprocessing.",
            )
        ]

    metadata = _read_json(image.sidecar)
    issues: list[ContractIssue] = []
    if image.modality is Modality.UNKNOWN:
        issues.append(
            _issue("error", "unknown_modality", image, "Acquisition needs manual classification.")
        )
        return issues

    if image.modality is Modality.FMRI:
        _require(metadata, ("RepetitionTime",), image, issues, "fmri_repetition_time")

    elif image.modality is Modality.DWI:
        for extension in (".bval", ".bvec"):
            if not _companion(image.path, extension).exists():
                issues.append(
                    _issue(
                        "error",
                        f"dwi_missing_{extension[1:]}",
                        image,
                        f"DWI requires a matching {extension} gradient file.",
                    )
                )
        _require(
            metadata,
            ("PhaseEncodingDirection",),
            image,
            issues,
            "dwi_phase_encoding",
            severity="warning",
        )
        if not any(metadata.get(field) is not None for field in ("TotalReadoutTime", "EffectiveEchoSpacing")):
            issues.append(
                _issue(
                    "warning",
                    "dwi_readout_timing",
                    image,
                    "Distortion correction needs TotalReadoutTime or EffectiveEchoSpacing.",
                )
            )

    elif image.modality is Modality.PET:
        _require(metadata, ("Units",), image, issues, "pet_units")
        _require_any(
            metadata,
            ("TracerName", "Radiopharmaceutical"),
            image,
            issues,
            "pet_tracer",
        )
        if _dimensions(image.path) == 4:
            _require(metadata, ("FrameTimesStart",), image, issues, "pet_frame_timing")
            _require(metadata, ("FrameDuration",), image, issues, "pet_frame_duration")

    elif image.modality is Modality.ASL:
        _require(
            metadata,
            ("ArterialSpinLabelingType", "M0Type"),
            image,
            issues,
            "asl_acquisition",
        )
        if not _companion(image.path, "_aslcontext.tsv", replace_suffix="_asl").exists():
            issues.append(
                _issue(
                    "error",
                    "asl_missing_context",
                    image,
                    "ASL requires a matching *_aslcontext.tsv file.",
                )
            )

    elif image.modality is Modality.SWI:
        _require(metadata, ("EchoTime",), image, issues, "swi_echo_time")
        _require(
            metadata,
            ("MagneticFieldStrength",),
            image,
            issues,
            "swi_field_strength",
            severity="warning",
        )

    elif image.modality is Modality.CT:
        units = str(metadata.get("Units", "")).upper()
        if units != "HU" and metadata.get("RescaleSlope") is None:
            issues.append(
                _issue(
                    "warning",
                    "ct_hu_scaling_unverified",
                    image,
                    "CT requires HU units or documented DICOM rescale-slope/intercept conversion.",
                )
            )

    elif image.modality is Modality.MRA:
        text = " ".join(
            str(metadata.get(field, ""))
            for field in ("SeriesDescription", "ProtocolName", "SequenceName")
        ).lower()
        if not any(token in text for token in ("mra", "angio", "tof", "phase contrast")):
            issues.append(
                _issue(
                    "warning",
                    "mra_subtype_uncertain",
                    image,
                    "The MRA subtype could not be confirmed from acquisition metadata.",
                )
            )

    elif image.modality is Modality.FIELDMAP:
        if not any(metadata.get(field) is not None for field in ("EchoTime", "EchoTime1", "EchoTime2")):
            issues.append(
                _issue(
                    "error",
                    "fieldmap_echo_time",
                    image,
                    "Field maps require EchoTime or EchoTime1/EchoTime2 metadata.",
                )
            )
        _require(
            metadata,
            ("IntendedFor",),
            image,
            issues,
            "fieldmap_intended_for",
            severity="warning",
        )

    return issues


def _audit_electrophysiology(image: ImageRecord) -> list[ContractIssue]:
    metadata = _read_json(image.sidecar)
    issues: list[ContractIssue] = []
    _require(metadata, ("SamplingFrequency",), image, issues, "electrophysiology_sampling_frequency")
    _require(
        metadata,
        ("PowerLineFrequency",),
        image,
        issues,
        "electrophysiology_power_line_frequency",
        severity="warning",
    )
    stem = image.path.name
    for suffix in (".edf", ".bdf", ".vhdr", ".set", ".fif", ".con", ".sqd", ".nwb", ".mefd", ".ds"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for suffix in ("_eeg", "_meg", "_ieeg"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    channels = image.path.parent / f"{stem}_channels.tsv"
    if not channels.exists():
        issues.append(
            _issue(
                "error",
                "electrophysiology_missing_channels",
                image,
                "BIDS EEG/MEG/iEEG recordings require a matching channels.tsv file.",
            )
        )
    if image.modality is Modality.IEEG:
        for suffix, code in (
            ("electrodes.tsv", "ieeg_missing_electrodes"),
            ("coordsystem.json", "ieeg_missing_coordsystem"),
        ):
            companion = image.path.parent / f"{stem}_{suffix}"
            if not companion.exists():
                issues.append(
                    _issue(
                        "error",
                        code,
                        image,
                        f"BIDS iEEG requires a matching {suffix} file.",
                    )
                )
    return issues


def _dimensions(path: Path) -> int:
    try:
        return len(nib.load(str(path)).shape)
    except Exception:
        return 0


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _companion(path: Path, extension: str, replace_suffix: str | None = None) -> Path:
    stem = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    if replace_suffix and stem.endswith(replace_suffix):
        stem = stem[: -len(replace_suffix)]
    return path.parent / f"{stem}{extension}"


def _require(
    metadata: dict[str, Any],
    fields: tuple[str, ...],
    image: ImageRecord,
    issues: list[ContractIssue],
    code: str,
    severity: str = "error",
) -> None:
    missing = [field for field in fields if metadata.get(field) in (None, "", [])]
    if missing:
        issues.append(
            _issue(
                severity,
                code,
                image,
                f"Missing required metadata: {', '.join(missing)}.",
            )
        )


def _require_any(
    metadata: dict[str, Any],
    fields: tuple[str, ...],
    image: ImageRecord,
    issues: list[ContractIssue],
    code: str,
) -> None:
    if not any(metadata.get(field) not in (None, "", []) for field in fields):
        issues.append(
            _issue("error", code, image, f"One of {', '.join(fields)} is required.")
        )


def _issue(
    severity: str,
    code: str,
    image: ImageRecord,
    message: str,
) -> ContractIssue:
    return ContractIssue(severity, code, str(image.path), message)
