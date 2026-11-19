from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np

from .visual_qc import write_image_qc_report


NIFTI_EXTENSIONS = (".nii.gz", ".nii")
ENTITY_PATTERN = re.compile(r"(?:^|_)([A-Za-z0-9]+)-([^_]+)")
SITE_METADATA_FIELDS = (
    "InstitutionName",
    "InstitutionAddress",
    "Manufacturer",
    "ManufacturersModelName",
    "DeviceSerialNumber",
    "StationName",
    "MagneticFieldStrength",
    "SoftwareVersions",
)


@dataclass(frozen=True)
class FinalizationPolicy:
    modality: str
    pipeline: str
    datatype: str
    payload_patterns: tuple[str, ...]
    payload_kind: str
    allowed_dimensions: tuple[int, ...]
    mask_required: bool = True
    apply_mask: bool = True
    required_auxiliaries: tuple[str, ...] = ()
    required_metadata_any: tuple[tuple[str, ...], ...] = ()
    corrections: tuple[str, ...] = ()
    fill_value: float = 0.0
    quantitative: bool = False
    report_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


POLICIES: dict[str, FinalizationPolicy] = {
    "t2w": FinalizationPolicy(
        "t2w",
        "neuroprep-structural",
        "anat",
        ("*_desc-preproc_T2w.nii.gz",),
        "spatial-image",
        (3,),
        corrections=("denoising", "N4 bias-field correction", "anatomical alignment"),
        report_required=False,
    ),
    "flair": FinalizationPolicy(
        "flair",
        "neuroprep-structural",
        "anat",
        ("*_desc-preproc_FLAIR.nii.gz",),
        "spatial-image",
        (3,),
        corrections=("denoising", "N4 bias-field correction", "anatomical alignment"),
        report_required=False,
    ),
    "pd": FinalizationPolicy(
        "pd",
        "neuroprep-structural",
        "anat",
        ("*_desc-preproc_PD.nii.gz",),
        "spatial-image",
        (3,),
        corrections=("denoising", "N4 bias-field correction", "anatomical alignment"),
        report_required=False,
    ),
    "fmri": FinalizationPolicy(
        "fmri",
        "xcpd",
        "func",
        ("*_desc-denoised_bold.nii.gz",),
        "spatiotemporal-image",
        (4,),
        required_auxiliaries=("outliers", "linc-qc"),
        required_metadata_any=(("RepetitionTime",),),
        corrections=(
            "fMRIPrep slice-timing, motion, distortion, and anatomical correction",
            "XCP-D despiking",
            "XCP-D nuisance regression",
            "XCP-D band-pass filtering",
            "XCP-D high-motion volume censoring",
        ),
    ),
    "dwi": FinalizationPolicy(
        "dwi",
        "qsiprep",
        "dwi",
        ("*_desc-preproc_dwi.nii.gz",),
        "gradient-image",
        (4,),
        required_auxiliaries=("bval", "bvec", "confounds"),
        corrections=(
            "thermal-noise removal when supported",
            "Gibbs-ringing correction when supported",
            "head-motion and eddy-current correction",
            "susceptibility-distortion correction when supported",
            "bias-field correction",
            "gradient rotation",
        ),
    ),
    "pet": FinalizationPolicy(
        "pet",
        "petprep",
        "pet",
        ("*_desc-preproc_pet.nii.gz",),
        "quantitative-spatiotemporal-image",
        (3, 4),
        required_auxiliaries=("confounds-if-4d",),
        required_metadata_any=(
            ("Units",),
            ("TracerName", "Radiopharmaceutical"),
        ),
        corrections=("framewise motion correction", "anatomical alignment"),
        quantitative=True,
        report_required=False,
    ),
    "asl": FinalizationPolicy(
        "asl",
        "aslprep",
        "perf",
        ("*_desc-preproc_asl.nii.gz", "*_desc-cbf.nii.gz", "*_cbf.nii.gz"),
        "quantitative-image",
        (3, 4),
        corrections=(
            "motion correction",
            "susceptibility-distortion correction when supported",
            "M0 calibration",
            "CBF quantification",
            "anatomical and standard-space alignment",
        ),
        quantitative=True,
        report_required=False,
    ),
    "swi": FinalizationPolicy(
        "swi",
        "neuroprep-susceptibility",
        "anat",
        ("*_Chimap.nii.gz", "*_qsm.nii.gz", "*_QSM.nii.gz", "*_swi.nii.gz"),
        "quantitative-image",
        (3,),
        corrections=(
            "phase unwrapping",
            "echo combination when applicable",
            "background-field removal",
            "dipole inversion",
            "QSM referencing",
        ),
        quantitative=True,
        report_required=False,
    ),
    "mra": FinalizationPolicy(
        "mra",
        "neuroprep-mra",
        "anat",
        ("*_desc-preproc_angio.nii.gz", "*_desc-preproc_MRA.nii.gz"),
        "spatial-image",
        (3,),
        corrections=("bias-field correction when applicable", "anatomical alignment"),
        report_required=False,
    ),
    "ct": FinalizationPolicy(
        "ct",
        "neuroprep-ct",
        "ct",
        ("*_desc-preproc_CT.nii.gz", "*_desc-preproc_ct.nii.gz"),
        "quantitative-image",
        (3,),
        required_metadata_any=(("Units",),),
        corrections=("DICOM rescale-slope/intercept application", "rigid anatomical alignment"),
        fill_value=-1024.0,
        quantitative=True,
        report_required=False,
    ),
    "eeg": FinalizationPolicy(
        "eeg",
        "mne-bids-pipeline-eeg",
        "eeg",
        ("*_proc-clean_raw.fif", "*_proc-clean_epo.fif", "*_proc-clean_epo.fif.gz"),
        "sensor-timeseries",
        (),
        mask_required=False,
        apply_mask=False,
        required_auxiliaries=("channels", "events-if-epochs"),
        corrections=("filtering", "bad-channel handling", "ICA or SSP artifact correction"),
    ),
    "meg": FinalizationPolicy(
        "meg",
        "mne-bids-pipeline-meg",
        "meg",
        ("*_proc-clean_raw.fif", "*_proc-clean_epo.fif", "*_proc-clean_epo.fif.gz"),
        "sensor-timeseries",
        (),
        mask_required=False,
        apply_mask=False,
        required_auxiliaries=("channels", "events-if-epochs"),
        corrections=("Maxwell filtering when configured", "bad-channel handling", "ICA or SSP artifact correction"),
    ),
    "ieeg": FinalizationPolicy(
        "ieeg",
        "mne-bids-pipeline-ieeg",
        "ieeg",
        ("*_proc-clean_raw.fif", "*_proc-clean_epo.fif", "*_proc-clean_epo.fif.gz"),
        "sensor-timeseries",
        (),
        mask_required=False,
        apply_mask=False,
        required_auxiliaries=("channels", "electrodes", "coordsystem", "events-if-epochs"),
        corrections=("filtering", "bad-channel handling", "ICA or SSP artifact correction"),
    ),
}


FMRI_PIPELINE_POLICIES: dict[str, FinalizationPolicy] = {
    "deepprep": FinalizationPolicy(
        "fmri",
        "deepprep",
        "func",
        ("*_desc-preproc_bold.nii.gz",),
        "spatiotemporal-image",
        (4,),
        required_auxiliaries=("confounds",),
        required_metadata_any=(("RepetitionTime",),),
        corrections=(
            "DeepPrep slice-timing correction",
            "DeepPrep motion correction",
            "DeepPrep susceptibility-distortion correction when supported",
            "DeepPrep anatomical and standard-space alignment",
            "DeepPrep confound estimation",
        ),
    ),
    "deepprep-postprocess": FinalizationPolicy(
        "fmri",
        "deepprep-postprocess",
        "func",
        (
            "*_desc-bandpass_bold.nii.gz",
            "*_desc-denoised_bold.nii.gz",
            "*_desc-clean_bold.nii.gz",
            "*_desc-preproc_bold.nii.gz",
        ),
        "spatiotemporal-image",
        (4,),
        required_auxiliaries=("confounds",),
        required_metadata_any=(("RepetitionTime",),),
        corrections=(
            "DeepPrep slice-timing, motion, distortion, and anatomical correction",
            "DeepPrep resting-state band-pass filtering",
            "DeepPrep nuisance regression",
            "high-motion metrics retained for censoring and QC",
        ),
    ),
}


def finalization_policy(
    modality: str, source_pipeline: str | None = None
) -> FinalizationPolicy:
    key = modality.lower()
    if key == "fmri" and source_pipeline in FMRI_PIPELINE_POLICIES:
        return FMRI_PIPELINE_POLICIES[str(source_pipeline)]
    try:
        return POLICIES[key]
    except KeyError as exc:
        raise ValueError(f"No model-ready finalization policy for modality: {modality}") from exc


def finalize_modality_subject(
    pipeline_root: Path,
    output_root: Path,
    bids_root: Path,
    subject: str,
    modality: str,
    marker: Path,
    required_spaces: tuple[str, ...] = (),
    support_roots: tuple[Path, ...] = (),
    source_pipeline: str | None = None,
) -> dict[str, Any]:
    """Validate and materialize one modality's model-ready payloads.

    The function never estimates transforms or alters modality physics. It consumes outputs from
    the reviewed upstream expert, applies an already-estimated brain mask where appropriate, and
    fails closed when required metadata or companion data are absent.
    """

    policy = finalization_policy(modality, source_pipeline)
    subject_token = subject.removeprefix("sub-")
    subject_root = _find_subject_root(pipeline_root, subject_token)
    support_subject_roots = [
        _find_subject_root(root, subject_token) for root in support_roots
    ]
    reports = _find_subject_reports(pipeline_root, subject_root, subject_token)
    for root, support_subject_root in zip(
        support_roots, support_subject_roots, strict=True
    ):
        reports.extend(
            _find_subject_reports(root, support_subject_root, subject_token)
        )
    reports = sorted(set(reports))
    if policy.report_required and not reports:
        raise FileNotFoundError(
            f"No upstream visual QC report for sub-{subject_token} under {pipeline_root}"
        )
    payloads = _discover_payloads(subject_root, policy)
    if not payloads:
        exclusions = _discover_upstream_exclusions(subject_root, policy.modality)
        if exclusions:
            summary = {
                "status": "pass",
                "subject": subject_token,
                "modality": policy.modality,
                "pipeline": policy.pipeline,
                "policy": policy.to_dict(),
                "pipeline_root": str(pipeline_root),
                "support_roots": [str(path) for path in support_roots],
                "output_root": str(output_root),
                "required_spaces": list(required_spaces),
                "observed_spaces": [],
                "visual_reports": [str(path) for path in reports],
                "records": [],
                "excluded_records": exclusions,
                "failures": [],
                "disposition": "handled-nonvolumetric-acquisition",
            }
            _write_json(marker, summary)
            return summary
        raise FileNotFoundError(
            f"No {policy.modality} payload matching {policy.payload_patterns} under {subject_root}"
        )

    records: list[dict[str, Any]] = []
    failures: list[str] = []
    observed_spaces: set[str] = set()
    for payload in payloads:
        try:
            record = (
                _finalize_sensor(payload, output_root, bids_root, subject_token, policy)
                if policy.payload_kind == "sensor-timeseries"
                else _finalize_image(
                    payload,
                    output_root,
                    subject_token,
                    policy,
                    subject_root,
                    support_subject_roots,
                )
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            failures.append(f"{payload.name}:{exc}")
            continue
        generated_report = record.pop("generated_visual_report", None)
        record["visual_reports"] = [
            *([str(generated_report)] if generated_report else []),
            *[str(path) for path in reports],
        ]
        records.append(record)
        observed_spaces.add(str(record["space"]))
        failures.extend(f"{payload.name}:{flag}" for flag in record["qc"]["flags"])

    for required in required_spaces:
        if not any(_space_matches(observed, required) for observed in observed_spaces):
            failures.append(f"missing-required-space:{required}")
    summary = {
        "status": "pass" if records and not failures else "fail",
        "subject": subject_token,
        "modality": policy.modality,
        "pipeline": policy.pipeline,
        "policy": policy.to_dict(),
        "pipeline_root": str(pipeline_root),
        "support_roots": [str(path) for path in support_roots],
        "output_root": str(output_root),
        "required_spaces": list(required_spaces),
        "observed_spaces": sorted(observed_spaces),
        "visual_reports": [str(path) for path in reports],
        "records": records,
        "failures": sorted(set(failures)),
    }
    _write_json(marker, summary)
    return summary


def _find_subject_root(root: Path, subject: str) -> Path:
    direct = root / f"sub-{subject}"
    if direct.exists():
        return direct
    candidates = sorted(path for path in root.rglob("sub-" + subject) if path.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    if not root.exists():
        raise FileNotFoundError(f"Pipeline derivative root does not exist: {root}")
    raise FileNotFoundError(f"No unique sub-{subject} derivative directory under {root}")


def _discover_payloads(subject_root: Path, policy: FinalizationPolicy) -> list[Path]:
    matches: set[Path] = set()
    for pattern in policy.payload_patterns:
        matches.update(path for path in subject_root.rglob(pattern) if path.is_file())
    return sorted(matches)


def _discover_upstream_exclusions(
    subject_root: Path, modality: str
) -> list[dict[str, Any]]:
    exclusions: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(subject_root.rglob("*executor.json")):
        try:
            payload = _read_json(path)
        except (OSError, ValueError):
            continue
        if str(payload.get("modality", "")).lower() != modality.lower():
            continue
        for record in payload.get("excluded_records", []):
            if not isinstance(record, dict):
                continue
            output = str(record.get("output", ""))
            reason = str(record.get("reason", "upstream-acquisition-excluded"))
            if output and not Path(output).is_file():
                continue
            public = dict(record)
            public["source_marker"] = str(path)
            exclusions[(output, reason)] = public
    return [
        exclusions[key]
        for key in sorted(exclusions)
    ]


def _find_subject_reports(root: Path, subject_root: Path, subject: str) -> list[Path]:
    reports = set(root.glob(f"sub-{subject}*.html"))
    reports.update(root.rglob(f"*sub-{subject}*.html"))
    reports.update(subject_root.rglob("*.html"))
    reports.update(subject_root.rglob("*report*.html"))
    if not reports:
        reports.update(root.glob("QC/*.html"))
        reports.update(root.glob("QC/**/*.html"))
    return sorted(path for path in reports if path.is_file())


def _finalize_image(
    payload: Path,
    output_root: Path,
    subject: str,
    policy: FinalizationPolicy,
    subject_root: Path,
    support_subject_roots: list[Path],
) -> dict[str, Any]:
    image = nib.load(str(payload))
    if len(image.shape) not in policy.allowed_dimensions:
        raise ValueError(
            f"expected {policy.allowed_dimensions} dimensions, found {len(image.shape)}"
        )
    if any(int(value) < 2 for value in image.shape[:3]):
        raise ValueError(f"degenerate image grid {image.shape}")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
        raise ValueError("invalid image affine")

    search_roots = [subject_root, *support_subject_roots]
    metadata_path = _matching_metadata(payload, search_roots)
    if metadata_path is None:
        raise FileNotFoundError(
            f"matching JSON sidecar not found for {payload.name} across "
            f"{', '.join(str(root) for root in search_roots)}"
        )
    metadata = _read_json(metadata_path)
    for alternatives in policy.required_metadata_any:
        if not any(_metadata_present(metadata, field) for field in alternatives):
            raise ValueError(f"metadata requires one of {alternatives}")

    relative_directory = _release_relative_directory(payload, subject)
    output_directory = output_root / relative_directory
    output_name = _model_ready_name(payload.name)
    output_path = output_directory / output_name
    mask_path: Path | None = None
    mask_source_path: Path | None = None
    mask_image: nib.spatialimages.SpatialImage | None = None
    derived_mask = False
    if policy.mask_required:
        mask_path = _matching_mask(payload, search_roots)
        if mask_path is None:
            if policy.modality != "fmri":
                raise FileNotFoundError("matching desc-brain_mask NIfTI not found")
            mask_source_path = _matching_skull_stripped_anatomical(
                payload, search_roots
            )
            if mask_source_path is None:
                raise FileNotFoundError(
                    "matching desc-brain_mask or aligned skull-stripped T1w not found"
                )
            source_image = nib.load(str(mask_source_path))
            source_data = np.asarray(source_image.dataobj, dtype=np.float32)
            source_mask = np.isfinite(source_data) & (
                np.abs(source_data) > np.finfo(np.float32).eps
            )
            mask_header = source_image.header.copy()
            mask_header.set_data_dtype(np.uint8)
            mask_image = nib.Nifti1Image(
                source_mask.astype(np.uint8), source_image.affine, mask_header
            )
            mask_image.set_qform(
                source_image.get_qform(), int(source_image.header["qform_code"])
            )
            mask_image.set_sform(
                source_image.get_sform(), int(source_image.header["sform_code"])
            )
            mask_path = output_directory / _derived_brain_mask_name(payload.name)
            derived_mask = True
        else:
            mask_source_path = mask_path
            mask_image = nib.load(str(mask_path))
        if image.shape[:3] != mask_image.shape[:3]:
            raise ValueError(f"mask shape {mask_image.shape} does not match {image.shape[:3]}")
        if not np.allclose(image.affine, mask_image.affine, atol=1e-5):
            raise ValueError("mask affine does not match payload affine")

    auxiliaries = _required_image_auxiliaries(
        payload, search_roots, policy, len(image.shape)
    )

    data = np.asarray(image.dataobj, dtype=np.float32)
    flags: list[str] = []
    if not np.isfinite(data).all():
        flags.append("non-finite-input-voxels")
    mask_fraction: float | None = None
    outside_nonfill = 0
    if policy.apply_mask and mask_image is not None:
        mask_data = np.asanyarray(mask_image.dataobj, dtype=np.float32)
        mask = np.isfinite(mask_data) & (mask_data > 0.5)
        mask_fraction = float(mask.mean())
        if not mask.any():
            flags.append("empty-brain-mask")
        if mask.all():
            flags.append("brain-mask-covers-entire-grid")
        output_data = data
        if output_data.ndim == 3:
            output_data[~mask] = np.float32(policy.fill_value)
        else:
            output_data[~mask, :] = np.float32(policy.fill_value)
        outside = output_data[~mask] if output_data.ndim == 3 else output_data[~mask, :]
        outside_nonfill = int(np.count_nonzero(outside != np.float32(policy.fill_value)))
    else:
        output_data = data

    if not np.isfinite(output_data).all():
        flags.append("non-finite-output-voxels")
    if np.count_nonzero(output_data != np.float32(policy.fill_value)) == 0:
        flags.append("empty-model-ready-image")
    if outside_nonfill:
        flags.append("values-outside-brain-mask")

    output_directory.mkdir(parents=True, exist_ok=True)
    if derived_mask and mask_image is not None and mask_path is not None:
        _atomic_nifti(mask_image, mask_path)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    output_image = nib.Nifti1Image(output_data, image.affine, header)
    output_image.set_qform(image.get_qform(), int(image.header["qform_code"]))
    output_image.set_sform(image.get_sform(), int(image.header["sform_code"]))
    _atomic_nifti(output_image, output_path)

    flags.extend(_validate_auxiliary_alignment(auxiliaries, image.shape))
    flags.extend(_validate_quantitative_metadata(policy, metadata, image.shape))
    if policy.modality == "asl" and "cbf" in payload.name.lower():
        if not any(_metadata_present(metadata, key) for key in ("Units", "QuantitativeUnits")):
            flags.append("cbf-units-missing")

    copied_auxiliaries: dict[str, str] = {}
    for label, source in auxiliaries.items():
        destination = (
            _replace_nifti_extension(output_path, "." + label)
            if label in {"bval", "bvec"}
            else output_directory / source.name
        )
        _atomic_link_or_copy(source, destination)
        copied_auxiliaries[label] = str(destination)

    space = _entities(payload).get("space", "native")
    is_qsm = policy.modality == "swi" and any(
        token in payload.name.lower() for token in ("qsm", "chimap")
    )
    quantitative = policy.quantitative and (policy.modality != "swi" or is_qsm)
    corrections = (
        policy.corrections
        if policy.modality != "swi" or is_qsm
        else ("denoising", "N4 bias-field correction", "anatomical alignment")
    )
    output_metadata = {
        **metadata,
        "NeuroPrepContractVersion": "3.0",
        "ModelReady": not flags,
        "SkullStripped": bool(policy.apply_mask),
        "SpatialReference": space,
        "QuantitativeValuesPreserved": quantitative,
        "BackgroundFillValue": policy.fill_value if policy.apply_mask else None,
        "Sources": [
            str(payload),
            *([str(mask_source_path)] if mask_source_path else []),
        ],
        "BrainMaskDerivation": (
            "nonzero support of same-grid skull-stripped T1w"
            if derived_mask
            else "upstream explicit brain mask"
        )
        if policy.mask_required
        else None,
        "CorrectionsApplied": list(corrections),
        "GeneratedBy": [
            {
                "Name": "NeuroPrep modality finalizer",
                "Description": "Validated upstream outputs and applied the existing brain mask without resampling",
            }
        ],
    }
    output_metadata_path = _sidecar(output_path)
    _write_json(output_metadata_path, output_metadata)
    report_path = output_path.with_name(
        _split_extension(output_path.name)[0] + "_qc.html"
    )
    write_image_qc_report(
        output_path,
        mask_path,
        report_path,
        f"NeuroPrep {policy.modality.upper()} model-ready QC",
    )
    return {
        "modality": policy.modality,
        "payload_kind": "spatial-image" if policy.modality == "swi" and not is_qsm else policy.payload_kind,
        "space": space,
        "units": metadata.get("Units") or metadata.get("QuantitativeUnits"),
        "site_covariates": {
            field: metadata[field] for field in SITE_METADATA_FIELDS if field in metadata
        },
        "input": str(payload),
        "mask": str(mask_path) if mask_path else None,
        "output": str(output_path),
        "metadata": str(output_metadata_path),
        "auxiliaries": copied_auxiliaries,
        "sha256": _sha256(output_path),
        "generated_visual_report": str(report_path),
        "qc": {
            "status": "pass" if not flags else "fail",
            "flags": flags,
            "shape": [int(value) for value in output_data.shape],
            "voxel_sizes_mm": [float(value) for value in output_image.header.get_zooms()[:3]],
            "finite_fraction": float(np.isfinite(output_data).mean()),
            "brain_mask_fraction": mask_fraction,
            "values_outside_mask": outside_nonfill,
        },
    }


def _finalize_sensor(
    payload: Path,
    output_root: Path,
    bids_root: Path,
    subject: str,
    policy: FinalizationPolicy,
) -> dict[str, Any]:
    entities = _entities(payload)
    source_roots = [payload.parent, bids_root / f"sub-{subject}"]
    auxiliaries: dict[str, Path] = {}
    for label in policy.required_auxiliaries:
        if label == "events-if-epochs" and "_epo.fif" not in payload.name:
            continue
        candidate = _find_sensor_companion(source_roots, entities, label)
        if candidate is None:
            raise FileNotFoundError(f"required {label} companion not found")
        auxiliaries[label] = candidate

    output_directory = output_root / _release_relative_directory(payload, subject)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / _model_ready_name(payload.name)
    _atomic_link_or_copy(payload, output_path)
    copied: dict[str, str] = {}
    for label, source in auxiliaries.items():
        destination = output_directory / source.name
        _atomic_link_or_copy(source, destination)
        copied[label] = str(destination)
    return {
        "modality": policy.modality,
        "payload_kind": policy.payload_kind,
        "space": "sensor-native",
        "units": "recording-native",
        "site_covariates": {},
        "input": str(payload),
        "mask": None,
        "output": str(output_path),
        "metadata": None,
        "auxiliaries": copied,
        "sha256": _sha256(output_path),
        "qc": {
            "status": "pass",
            "flags": [],
            "bytes": output_path.stat().st_size,
            "cleaned": "proc-clean" in output_path.name,
        },
    }


def _matching_mask(payload: Path, subject_roots: list[Path]) -> Path | None:
    candidates = sorted(
        path
        for root in subject_roots
        for path in root.rglob("*_desc-brain_mask.nii.gz")
    )
    if not candidates:
        candidates = sorted(
            path for root in subject_roots for path in root.rglob("*_brain_mask.nii.gz")
        )
    if not candidates:
        candidates = sorted(
            path for root in subject_roots for path in root.rglob("*_mask.nii.gz")
        )
    payload_entities = _entities(payload)
    ranked: list[tuple[int, Path]] = []
    for candidate in candidates:
        candidate_entities = _entities(candidate)
        mandatory = ("sub", "ses", "task", "acq", "run", "echo", "space", "res")
        mismatch = any(
            not _entity_values_match(
                key, payload_entities.get(key), candidate_entities.get(key)
            )
            for key in mandatory
            if key in payload_entities and key in candidate_entities
        )
        missing_specific = any(
            key in payload_entities and key not in candidate_entities
            for key in ("sub", "ses", "task", "run", "space", "res")
        )
        extra_acquisition = any(
            key in candidate_entities and key not in payload_entities
            for key in ("task", "acq", "run", "echo")
        )
        if mismatch or missing_specific or extra_acquisition:
            continue
        score = sum(
            payload_entities.get(key) == candidate_entities.get(key)
            for key in mandatory
            if key in payload_entities
        )
        if candidate.parent == payload.parent:
            score += 5
        ranked.append((score, candidate))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1].as_posix()))
    best_score = ranked[0][0]
    best = [path for score, path in ranked if score == best_score]
    return best[0] if len(best) == 1 else None


def _matching_skull_stripped_anatomical(
    payload: Path, subject_roots: list[Path]
) -> Path | None:
    candidates = sorted(
        {
            path
            for root in subject_roots
            for pattern in (
                "*_desc-noskull_T1w.nii.gz",
                "*_desc-brain_T1w.nii.gz",
            )
            for path in root.rglob(pattern)
            if "WorkDir" not in path.parts
        }
    )
    payload_entities = _entities(payload)
    payload_session = payload_entities.get("ses")
    ranked: list[tuple[int, Path]] = []
    for candidate in candidates:
        candidate_entities = _entities(candidate)
        candidate_session = candidate_entities.get("ses") or next(
            (
                part.removeprefix("ses-")
                for part in candidate.parts
                if part.startswith("ses-")
            ),
            None,
        )
        if payload_session and candidate_session != payload_session:
            continue
        if any(
            not _entity_values_match(
                key, payload_entities.get(key), candidate_entities.get(key)
            )
            for key in ("sub", "space", "res")
            if key in payload_entities and key in candidate_entities
        ):
            continue
        score = sum(
            payload_entities.get(key) == candidate_entities.get(key)
            for key in ("sub", "space", "res")
            if key in payload_entities
        )
        score += int(candidate_session == payload_session)
        score += int("_desc-noskull_" in candidate.name)
        score += 2 * int("BOLD" in candidate.parts)
        ranked.append((score, candidate))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1].as_posix()))
    best_score = ranked[0][0]
    best = [path for score, path in ranked if score == best_score]
    return best[0] if len(best) == 1 else None


def _entity_values_match(key: str, left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return left == right
    if key == "res":
        try:
            return float(left) == float(right)
        except ValueError:
            pass
    return left == right


def _matching_metadata(payload: Path, subject_roots: list[Path]) -> Path | None:
    direct = _sidecar(payload)
    if direct.exists():
        return direct
    stem, _ = _split_extension(payload.name)
    suffix = stem.rsplit("_", 1)[-1]
    candidates = sorted(
        {
            path
            for root in subject_roots
            for path in root.rglob(f"*_{suffix}.json")
            if path.is_file()
        }
    )
    entities = _entities(payload)
    ranked: list[tuple[int, Path]] = []
    keys = ("sub", "ses", "task", "acq", "run", "echo", "space", "res")
    for candidate in candidates:
        candidate_entities = _entities(candidate)
        if any(
            entities.get(key) != candidate_entities.get(key)
            for key in keys
            if key in entities and key in candidate_entities
        ):
            continue
        score = sum(
            entities.get(key) == candidate_entities.get(key)
            for key in keys
            if key in entities
        )
        if candidate.parent == payload.parent:
            score += 5
        ranked.append((score, candidate))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1].as_posix()))
    best_score = ranked[0][0]
    best = [path for score, path in ranked if score == best_score]
    return best[0] if len(best) == 1 else None


def _required_image_auxiliaries(
    payload: Path,
    subject_roots: list[Path],
    policy: FinalizationPolicy,
    dimensions: int,
) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for label in policy.required_auxiliaries:
        if label == "confounds-if-4d" and dimensions != 4:
            continue
        normalized = "confounds" if label == "confounds-if-4d" else label
        if normalized in {"bval", "bvec"}:
            candidate = _replace_nifti_extension(payload, "." + normalized)
        elif normalized == "confounds":
            candidate = _matching_confounds(payload, subject_roots)
        elif normalized == "outliers":
            candidate = _matching_table(payload, subject_roots, ("*_outliers.tsv",))
        elif normalized == "linc-qc":
            candidate = _matching_table(
                payload, subject_roots, ("*_desc-linc_qc.tsv",)
            )
        else:
            raise ValueError(f"unknown auxiliary rule {label}")
        if candidate is None or not candidate.exists():
            raise FileNotFoundError(f"required {normalized} companion not found")
        found[normalized] = candidate
    return found


def _matching_confounds(payload: Path, subject_roots: list[Path]) -> Path | None:
    candidates = sorted(
        {
            path
            for root in subject_roots
            for path in root.rglob("*_desc-confounds_timeseries.tsv")
        }
    )
    entities = _entities(payload)
    ranked: list[tuple[int, Path]] = []
    for candidate in candidates:
        candidate_entities = _entities(candidate)
        keys = ("sub", "ses", "task", "acq", "run", "echo")
        if any(
            entities.get(key) != candidate_entities.get(key)
            for key in keys
            if key in entities and key in candidate_entities
        ):
            continue
        score = sum(
            entities.get(key) == candidate_entities.get(key)
            for key in keys
            if key in entities
        )
        if candidate.parent == payload.parent:
            score += 5
        ranked.append((score, candidate))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1].as_posix()))
    best_score = ranked[0][0]
    best = [path for score, path in ranked if score == best_score]
    return best[0] if len(best) == 1 else None


def _matching_table(
    payload: Path, subject_roots: list[Path], patterns: tuple[str, ...]
) -> Path | None:
    candidates = sorted(
        {
            path
            for root in subject_roots
            for pattern in patterns
            for path in root.rglob(pattern)
        }
    )
    entities = _entities(payload)
    ranked: list[tuple[int, Path]] = []
    for candidate in candidates:
        candidate_entities = _entities(candidate)
        keys = ("sub", "ses", "task", "acq", "run", "space")
        if any(
            entities.get(key) != candidate_entities.get(key)
            for key in keys
            if key in entities and key in candidate_entities
        ):
            continue
        score = sum(
            entities.get(key) == candidate_entities.get(key)
            for key in keys
            if key in entities
        )
        if candidate.parent == payload.parent:
            score += 5
        ranked.append((score, candidate))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1].as_posix()))
    best_score = ranked[0][0]
    best = [path for score, path in ranked if score == best_score]
    return best[0] if len(best) == 1 else None


def _find_sensor_companion(
    roots: Iterable[Path], entities: dict[str, str], label: str
) -> Path | None:
    suffixes = {
        "channels": ("_channels.tsv",),
        "events": ("_events.tsv",),
        "electrodes": ("_electrodes.tsv",),
        "coordsystem": ("_coordsystem.json",),
    }
    patterns = suffixes.get(label, suffixes.get(label.removesuffix("-if-epochs")))
    if patterns is None:
        return None
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            for suffix in patterns:
                candidates.extend(root.rglob("*" + suffix))
    subject = entities.get("sub")
    ranked: list[tuple[int, Path]] = []
    for path in sorted(set(candidates)):
        candidate_entities = _entities(path)
        if subject is not None and candidate_entities.get("sub") != subject:
            continue
        keys = ("sub", "ses", "task", "acq", "run")
        if any(
            entities.get(key) != candidate_entities.get(key)
            for key in keys
            if key in entities and key in candidate_entities
        ):
            continue
        score = sum(
            entities.get(key) == candidate_entities.get(key)
            for key in keys
            if key in entities
        )
        ranked.append((score, path))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1].as_posix()))
    best_score = ranked[0][0]
    best = [path for score, path in ranked if score == best_score]
    return best[0] if len(best) == 1 else None


def _release_relative_directory(payload: Path, subject: str) -> Path:
    parts = payload.parts
    session = next((part for part in parts if part.startswith("ses-")), None)
    datatype = payload.parent.name if payload.parent.name in {
        "anat", "func", "dwi", "pet", "perf", "ct", "eeg", "meg", "ieeg"
    } else "misc"
    return Path(f"sub-{subject}") / (session or "") / datatype


def _model_ready_name(name: str) -> str:
    if "_desc-preproc_" in name:
        return name.replace("_desc-preproc_", "_desc-modelready_", 1)
    if "_proc-clean_" in name:
        return name.replace("_proc-clean_", "_desc-modelready_", 1)
    if "_desc-" in name:
        return name.replace("_desc-", "_desc-modelready_", 1)
    stem, extension = _split_extension(name)
    if "_" in stem:
        prefix, suffix = stem.rsplit("_", 1)
        return f"{prefix}_desc-modelready_{suffix}{extension}"
    return f"{stem}_desc-modelready{extension}"


def _derived_brain_mask_name(name: str) -> str:
    stem, extension = _split_extension(name)
    prefix = re.sub(r"_desc-[^_]+_bold$", "", stem)
    if prefix == stem and stem.endswith("_bold"):
        prefix = stem.removesuffix("_bold")
    return f"{prefix}_desc-brain_mask{extension}"


def _validate_auxiliary_alignment(
    auxiliaries: dict[str, Path], shape: tuple[int, ...]
) -> list[str]:
    flags: list[str] = []
    volumes = int(shape[3]) if len(shape) == 4 else 1
    if "bval" in auxiliaries:
        try:
            bvals = np.loadtxt(auxiliaries["bval"], dtype=float).reshape(-1)
            if bvals.size != volumes:
                flags.append(f"bval-count-{bvals.size}-does-not-match-volumes-{volumes}")
        except (OSError, ValueError):
            flags.append("invalid-bval-file")
    if "bvec" in auxiliaries:
        try:
            bvecs = np.loadtxt(auxiliaries["bvec"], dtype=float)
            if bvecs.ndim == 1:
                bvecs = bvecs.reshape(3, -1)
            if bvecs.shape not in {(3, volumes), (volumes, 3)}:
                flags.append(f"bvec-shape-{tuple(bvecs.shape)}-does-not-match-volumes-{volumes}")
            if not np.isfinite(bvecs).all():
                flags.append("non-finite-bvec-values")
        except (OSError, ValueError):
            flags.append("invalid-bvec-file")
    if "confounds" in auxiliaries:
        try:
            with auxiliaries["confounds"].open("r", encoding="utf-8") as stream:
                rows = sum(1 for line in stream if line.strip()) - 1
            if rows != volumes:
                flags.append(f"confounds-rows-{rows}-does-not-match-volumes-{volumes}")
        except OSError:
            flags.append("invalid-confounds-file")
    return flags


def _validate_quantitative_metadata(
    policy: FinalizationPolicy, metadata: dict[str, Any], shape: tuple[int, ...]
) -> list[str]:
    flags: list[str] = []
    if policy.modality == "ct":
        units = str(metadata.get("Units", "")).lower().replace(" ", "")
        if units not in {"hu", "hounsfield", "hounsfieldunit", "hounsfieldunits"}:
            flags.append("ct-units-are-not-hounsfield-units")
    if policy.modality == "pet" and len(shape) == 4:
        volumes = int(shape[3])
        for key in ("FrameTimesStart", "FrameDuration"):
            values = metadata.get(key)
            if not isinstance(values, list) or len(values) != volumes:
                flags.append(f"{key}-count-does-not-match-volumes-{volumes}")
    return flags


def _entities(path: Path) -> dict[str, str]:
    stem, _ = _split_extension(path.name)
    entities = {key.lower(): value for key, value in ENTITY_PATTERN.findall(stem)}
    if "ses" not in entities:
        session = next(
            (part.removeprefix("ses-") for part in reversed(path.parts) if part.startswith("ses-")),
            None,
        )
        if session:
            entities["ses"] = session
    return entities


def _space_matches(observed: str, required: str) -> bool:
    return observed == required or observed.startswith(required + ":") or required in observed


def _metadata_present(metadata: dict[str, Any], key: str) -> bool:
    return key in metadata and metadata[key] not in (None, "", [], {})


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON sidecar {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON sidecar must contain an object: {path}")
    return payload


def _sidecar(path: Path) -> Path:
    return _replace_nifti_extension(path, ".json")


def _replace_nifti_extension(path: Path, extension: str) -> Path:
    name = path.name
    for candidate in NIFTI_EXTENSIONS:
        if name.endswith(candidate):
            return path.with_name(name[: -len(candidate)] + extension)
    return path.with_suffix(extension)


def _split_extension(name: str) -> tuple[str, str]:
    for extension in (".nii.gz", ".fif.gz", ".nii", ".fif"):
        if name.endswith(extension):
            return name[: -len(extension)], extension
    path = Path(name)
    return path.stem, path.suffix


def _atomic_nifti(image: nib.Nifti1Image, path: Path) -> None:
    temporary = path.parent / f".{path.name}.tmp.nii.gz"
    nib.save(image, str(temporary))
    temporary.replace(path)


def _atomic_link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() and source.resolve() == destination.resolve():
        return
    temporary = destination.parent / f".{destination.name}.tmp"
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(destination)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
