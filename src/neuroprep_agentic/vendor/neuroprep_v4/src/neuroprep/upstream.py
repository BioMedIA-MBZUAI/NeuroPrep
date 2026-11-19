from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

STRUCTURAL_SUFFIXES = {
    "t2w": "T2w",
    "flair": "FLAIR",
    "pd": "PD",
    "mra": "angio",
    "ct": "CT",
    "swi": "swi",
}


@dataclass(frozen=True)
class SpatialExecutorPolicy:
    modality: str
    denoise: bool
    n4_bias_correction: bool
    skull_strip: bool
    registration_transform: str
    interpolation: str
    background_fill: float
    preserve_quantitative_values: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SPATIAL_POLICIES = {
    "t2w": SpatialExecutorPolicy("t2w", True, True, True, "s", "LanczosWindowedSinc", 0.0, False),
    "flair": SpatialExecutorPolicy(
        "flair", True, True, True, "s", "LanczosWindowedSinc", 0.0, False
    ),
    "pd": SpatialExecutorPolicy("pd", True, True, True, "s", "LanczosWindowedSinc", 0.0, False),
    "mra": SpatialExecutorPolicy("mra", True, True, True, "s", "LanczosWindowedSinc", 0.0, False),
    # CT uses affine-only normalization by default to avoid pathology-erasing nonlinear warps.
    "ct": SpatialExecutorPolicy("ct", False, False, True, "a", "Linear", -1024.0, True),
    "swi": SpatialExecutorPolicy("swi", True, True, True, "s", "LanczosWindowedSinc", 0.0, False),
}


def preprocess_spatial_modalities(
    bids_root: Path,
    output_root: Path,
    work_root: Path,
    subject: str,
    modalities: tuple[str, ...],
    reference_root: Path | None = None,
    template: Path | None = None,
    execute: bool = True,
) -> dict[str, Any]:
    subject_token = subject.removeprefix("sub-")
    records = []
    failures = []
    for modality in modalities:
        inputs = _discover_spatial_inputs(
            bids_root, subject_token, modality.lower(), None
        )
        sessions = sorted(
            {
                session
                for source in inputs
                if (session := _session_entity(source)) is not None
            }
        )
        scopes: list[str | None] = sessions or [None]
        for session in scopes:
            try:
                reference = _find_subject_reference(
                    reference_root, subject_token, session
                )
                records.append(
                    preprocess_spatial_subject(
                        bids_root=bids_root,
                        output_root=output_root,
                        work_root=work_root,
                        subject=subject_token,
                        modality=modality,
                        reference=reference,
                        template=template,
                        session=session,
                        execute=execute,
                    )
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                label = f"{modality}:{session or 'single'}"
                failures.append(f"{label}:{exc}")
    summary = {
        "status": "pass" if records and not failures else "fail",
        "subject": subject_token,
        "modalities": list(modalities),
        "reference_policy": "explicit-standard-template-then-visit-matched-subject-anatomy",
        "records": records,
        "failures": failures,
    }
    _write_json(output_root / f"sub-{subject_token}" / "multimodal-executor.json", summary)
    return summary


def preprocess_spatial_subject(
    bids_root: Path,
    output_root: Path,
    work_root: Path,
    subject: str,
    modality: str,
    reference: Path | None = None,
    template: Path | None = None,
    session: str | None = None,
    execute: bool = True,
) -> dict[str, Any]:
    """Run a conservative complete anatomical/CT/MRA executor.

    All transforms are estimated before one final interpolation into the requested
    reference grid. Raw DWI is deliberately not accepted by this executor.
    """

    key = modality.lower()
    if key not in SPATIAL_POLICIES:
        raise ValueError(f"No spatial executor policy for modality: {modality}")
    policy = SPATIAL_POLICIES[key]
    subject_token = subject.removeprefix("sub-")
    inputs = _discover_spatial_inputs(bids_root, subject_token, key, session)
    if not inputs:
        raise FileNotFoundError(
            f"No BIDS {key} inputs found for sub-{subject_token} under {bids_root}"
        )
    projections: list[Path] = []
    if key == "mra":
        inputs, projections = _partition_mra_inputs(inputs)
    fixed = _resolve_reference(reference, template)
    if inputs and fixed is None:
        raise FileNotFoundError(
            "A subject anatomical reference or explicit standard-space template is required; "
            "set --reference or --template (NEUROPREP_MNI_TEMPLATE is also supported)."
        )
    if fixed is not None:
        _validate_nifti(fixed, "registration reference")

    records: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []
    if projections:
        excluded_records.extend(
            _prepare_mra_projection(
                source,
                output_root,
                subject_token,
                execute,
            )
            for source in projections
        )
    if inputs:
        assert fixed is not None
    for source in inputs:
        records.append(
            _preprocess_spatial_image(
                source,
                fixed,
                output_root,
                work_root,
                subject_token,
                policy,
                execute,
            )
        )
    summary = {
        "status": (
            "pass"
            if execute and (records or excluded_records)
            else "planned"
            if not execute and (records or excluded_records)
            else "fail"
        ),
        "subject": subject_token,
        "modality": key,
        "reference": str(fixed) if fixed is not None else None,
        "policy": policy.to_dict(),
        "records": records,
        "excluded_records": excluded_records,
        "acquisition_triage": (
            {
                "volumetric_inputs": len(inputs),
                "projection_inputs": len(excluded_records),
                "policy": (
                    "prefer original volumetric angiography; retain derived MIP projections "
                    "as normalized 2D auxiliaries and exclude them from 3D registration and "
                    "common-grid channel stacking"
                ),
            }
            if key == "mra"
            else None
        ),
    }
    marker = output_root / f"sub-{subject_token}"
    if session:
        marker = marker / f"ses-{session.removeprefix('ses-')}"
    marker = marker / "executor.json"
    _write_json(marker, summary)
    return summary


def _partition_mra_inputs(inputs: list[Path]) -> tuple[list[Path], list[Path]]:
    """Separate trainable 3D angiography from scanner-derived projection views."""

    volumetric: list[Path] = []
    projections: list[Path] = []
    for source in inputs:
        image = nib.load(str(source))
        metadata = _read_sidecar(source)
        raw_image_type = metadata.get("ImageType", [])
        image_type = (
            " ".join(str(value).lower() for value in raw_image_type)
            if isinstance(raw_image_type, list)
            else str(raw_image_type).lower()
        )
        is_projection = (
            len(image.shape) < 3
            or any(int(value) <= 1 for value in image.shape[:3])
            or "projection" in image_type
            or re.search(r"(?:^|\\s)(?:mip|minip)(?:\\s|$)", image_type) is not None
        )
        (projections if is_projection else volumetric).append(source)
    return volumetric, projections


def _prepare_mra_projection(
    source: Path,
    output_root: Path,
    subject: str,
    execute: bool,
) -> dict[str, Any]:
    """Normalize a derived MRA projection without misrepresenting it as a 3D volume."""

    image = nib.load(str(source))
    if len(image.shape) not in (2, 3) or (
        len(image.shape) == 3 and sum(int(value) > 1 for value in image.shape) != 2
    ):
        raise ValueError(f"MRA projection requires exactly two non-singleton axes: {source}")
    destination_dir = output_root / _relative_subject_directory(source, subject, "anat")
    destination_dir.mkdir(parents=True, exist_ok=True)
    output = destination_dir / (
        f"{_bids_prefix(source)}_desc-normalized_projection.nii.gz"
    )
    metadata_output = _sidecar(output)
    normalization: dict[str, Any] = {
        "method": "finite robust percentile scaling",
        "lower_percentile": 1.0,
        "upper_percentile": 99.0,
        "output_range": [0.0, 1.0],
    }
    if execute:
        data = np.asarray(image.dataobj, dtype=np.float32)
        finite = np.isfinite(data)
        foreground = data[finite & (data != 0)]
        if foreground.size < 16:
            foreground = data[finite]
        if foreground.size < 16:
            raise ValueError(f"MRA projection has insufficient finite signal: {source}")
        low, high = np.percentile(foreground, (1.0, 99.0))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"MRA projection has degenerate intensity range: {source}")
        normalized = np.clip((data - low) / (high - low), 0.0, 1.0)
        normalized[~finite] = 0.0
        header = image.header.copy()
        header.set_data_dtype(np.float32)
        output_image = nib.Nifti1Image(
            normalized.astype(np.float32), image.affine, header
        )
        output_image.set_qform(image.get_qform(), int(image.header["qform_code"]))
        output_image.set_sform(image.get_sform(), int(image.header["sform_code"]))
        nib.save(output_image, str(output))
        metadata = _read_sidecar(source)
        metadata.update(
            {
                "NeuroPrepExecutorVersion": "3.1",
                "ModelReady": False,
                "ProjectionReady": True,
                "PayloadKind": "projection-image",
                "SkullStripped": False,
                "SpatiallyNormalized": False,
                "ExcludedFromVolumetricModel": True,
                "ExclusionReason": "derived-2d-mra-projection-no-volumetric-source",
                "IntensityNormalization": normalization,
                "Sources": [str(source)],
                "GeneratedAt": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_json(metadata_output, metadata)
    return {
        "status": "excluded",
        "modality": "mra",
        "payload_kind": "projection-image",
        "input": str(source),
        "output": str(output),
        "metadata": str(metadata_output),
        "reason": "derived-2d-mra-projection-no-volumetric-source",
        "excluded_from": [
            "3d-bias-field-correction",
            "3d-skull-stripping",
            "3d-registration",
            "common-grid-channel-stacking",
        ],
        "normalization": normalization,
        "shape": [int(value) for value in image.shape],
    }


def preprocess_sensor_subject(
    bids_root: Path,
    output_root: Path,
    subject: str,
    modality: str,
    session: str | None = None,
    low_frequency: float | None = None,
    high_frequency: float | None = None,
    resample_hz: float | None = None,
    apply_ica: bool = True,
    execute: bool = True,
) -> dict[str, Any]:
    """Preprocess EEG/MEG/iEEG with deterministic MNE operations and QC metadata."""

    key = modality.lower()
    if key not in {"eeg", "meg", "ieeg"}:
        raise ValueError("Sensor executor modality must be eeg, meg, or ieeg")
    try:
        import mne
    except ImportError as exc:
        if execute:
            raise RuntimeError(
                "Sensor preprocessing requires NeuroPrep's sensors extra: "
                "pip install 'neuroprep[sensors]'"
            ) from exc
        mne = None

    defaults = {
        "eeg": (0.5, 100.0, 250.0),
        "meg": (0.5, 100.0, 500.0),
        "ieeg": (0.5, 200.0, 500.0),
    }
    default_low, default_high, default_rate = defaults[key]
    low_frequency = default_low if low_frequency is None else low_frequency
    high_frequency = default_high if high_frequency is None else high_frequency
    resample_hz = default_rate if resample_hz is None else resample_hz
    subject_token = subject.removeprefix("sub-")
    inputs = _discover_sensor_inputs(bids_root, subject_token, key, session)
    if not inputs:
        raise FileNotFoundError(f"No BIDS {key} recordings found for sub-{subject_token}")

    records: list[dict[str, Any]] = []
    for source in inputs:
        destination_dir = output_root / _relative_subject_directory(source, subject_token, key)
        destination_dir.mkdir(parents=True, exist_ok=True)
        prefix = _bids_prefix(source)
        output = destination_dir / f"{prefix}_proc-clean_raw.fif"
        report_path = destination_dir / f"{prefix}_proc-clean_report.html"
        record: dict[str, Any] = {
            "input": str(source),
            "output": str(output),
            "report": str(report_path),
            "filter_hz": [low_frequency, high_frequency],
            "resample_hz": resample_hz,
            "ica_requested": apply_ica,
            "commands": ["MNE-Python deterministic sensor preprocessing"],
        }
        if not execute:
            record["status"] = "planned"
            records.append(record)
            continue

        raw = _read_raw(mne, source)
        raw.load_data()
        original_rate = float(raw.info["sfreq"])
        line_frequency = raw.info.get("line_freq")
        if line_frequency and float(line_frequency) < high_frequency:
            frequencies = np.arange(
                float(line_frequency), high_frequency + 1, float(line_frequency)
            )
            raw.notch_filter(frequencies, verbose="error")
        raw.filter(low_frequency, min(high_frequency, original_rate / 2 - 0.5), verbose="error")
        bads = _detect_bad_channels(raw)
        raw.info["bads"] = sorted(set(raw.info.get("bads", [])) | set(bads))
        if key == "eeg":
            raw.set_eeg_reference("average", projection=False, verbose="error")
        ica_excluded: list[int] = []
        ica_warning = None
        if apply_ica and raw.n_times / original_rate >= 60:
            try:
                ica_excluded = _apply_ica(mne, raw)
            except Exception as exc:  # acquisition-specific channel layouts can defeat ICA
                ica_warning = f"ICA skipped after validated failure: {type(exc).__name__}: {exc}"
        if resample_hz < float(raw.info["sfreq"]):
            raw.resample(resample_hz, npad="auto", verbose="error")
        raw.save(output, overwrite=True, split_size="2GB", verbose="error")
        copied = _copy_sensor_companions(source, destination_dir)
        record.update(
            {
                "status": "pass",
                "original_sfreq_hz": original_rate,
                "final_sfreq_hz": float(raw.info["sfreq"]),
                "channels": len(raw.ch_names),
                "bad_channels": list(raw.info["bads"]),
                "ica_excluded": ica_excluded,
                "warnings": [ica_warning] if ica_warning else [],
                "companions": copied,
            }
        )
        _write_sensor_report(report_path, record)
        records.append(record)

    summary = {
        "status": "pass"
        if execute and all(row["status"] == "pass" for row in records)
        else "planned",
        "subject": subject_token,
        "modality": key,
        "records": records,
    }
    _write_json(output_root / f"sub-{subject_token}" / "executor.json", summary)
    return summary


def preprocess_susceptibility_subject(
    bids_root: Path,
    output_root: Path,
    work_root: Path,
    subject: str,
    reference_root: Path | None = None,
    template: Path | None = None,
    execute: bool = True,
) -> dict[str, Any]:
    subject_token = subject.removeprefix("sub-")
    inputs = _discover_spatial_inputs(bids_root, subject_token, "swi", None)
    components = {_susceptibility_component(path) for path in inputs}
    if {"magnitude", "phase"} <= components:
        qsmxt = os.environ.get("NEUROPREP_QSMXT", "qsmxt")
        command = [
            qsmxt,
            "run",
            str(bids_root),
            str(output_root),
            "--include",
            f"sub-{subject_token}*",
        ]
        if execute:
            _run(command)
        summary = {
            "status": "pass" if execute else "planned",
            "subject": subject_token,
            "mode": "qsm-reconstruction",
            "components": sorted(components),
            "command": command,
            "output_root": str(output_root / "derivatives" / "qsmxt.rs"),
        }
    else:
        summary = preprocess_spatial_modalities(
            bids_root,
            output_root,
            work_root,
            subject_token,
            ("swi",),
            reference_root,
            template,
            execute,
        )
        summary["mode"] = "reconstructed-swi-preprocessing"
        summary["components"] = sorted(components)
    marker_root = (
        output_root / "logs"
        if summary.get("mode") == "qsm-reconstruction"
        else output_root / f"sub-{subject_token}"
    )
    _write_json(marker_root / f"sub-{subject_token}_susceptibility-executor.json", summary)
    return summary


def _preprocess_spatial_image(
    source: Path,
    fixed: Path,
    output_root: Path,
    work_root: Path,
    subject: str,
    policy: SpatialExecutorPolicy,
    execute: bool,
) -> dict[str, Any]:
    datatype = "ct" if policy.modality == "ct" else "anat"
    destination_dir = output_root / _relative_subject_directory(source, subject, datatype)
    destination_dir.mkdir(parents=True, exist_ok=True)
    working = work_root / f"sub-{subject}" / policy.modality / _bids_prefix(source)
    working.mkdir(parents=True, exist_ok=True)
    suffix = STRUCTURAL_SUFFIXES[policy.modality]
    space = _space_label(fixed)
    output = destination_dir / f"{_bids_prefix(source)}_space-{space}_desc-preproc_{suffix}.nii.gz"
    mask_output = destination_dir / f"{_bids_prefix(source)}_space-{space}_desc-brain_mask.nii.gz"
    denoised = working / "denoised.nii.gz"
    corrected = working / "biascorr.nii.gz"
    native_brain = working / "brain.nii.gz"
    native_mask = working / "brain_mask.nii.gz"
    transform_prefix = working / "to-reference_"
    transform_stem = f"{_bids_prefix(source)}_from-{policy.modality}_to-{space}_mode-image"
    current = source
    commands: list[list[str]] = []
    if policy.denoise:
        commands.append(["DenoiseImage", "-d", "3", "-i", str(current), "-o", str(denoised)])
        current = denoised
    if policy.n4_bias_correction:
        commands.append(
            [
                "N4BiasFieldCorrection",
                "-d",
                "3",
                "-i",
                str(current),
                "-o",
                str(corrected),
            ]
        )
        current = corrected
    commands.append(
        [
            "mri_synthstrip",
            "-i",
            str(current),
            "-o",
            str(native_brain),
            "-m",
            str(native_mask),
        ]
    )
    commands.append(
        [
            "antsRegistrationSyNQuick.sh",
            "-d",
            "3",
            "-f",
            str(fixed),
            "-m",
            str(current),
            "-t",
            policy.registration_transform,
            "-o",
            str(transform_prefix),
        ]
    )
    transforms = _expected_transforms(transform_prefix, policy.registration_transform)
    transform_outputs = [
        destination_dir
        / (
            f"{transform_stem}_desc-warp_xfm.nii.gz"
            if transform.name.endswith("Warp.nii.gz")
            else f"{transform_stem}_desc-affine_xfm.mat"
        )
        for transform in transforms
    ]
    commands.append(
        [
            "antsApplyTransforms",
            "-d",
            "3",
            "-i",
            str(current),
            "-r",
            str(fixed),
            "-o",
            str(output),
            "-n",
            policy.interpolation,
            *[item for transform in transforms for item in ("-t", str(transform))],
        ]
    )
    commands.append(
        [
            "antsApplyTransforms",
            "-d",
            "3",
            "-i",
            str(native_mask),
            "-r",
            str(fixed),
            "-o",
            str(mask_output),
            "-n",
            "NearestNeighbor",
            *[item for transform in transforms for item in ("-t", str(transform))],
        ]
    )

    if execute:
        for command in commands:
            _run(command)
        for transform, derivative_transform in zip(transforms, transform_outputs, strict=True):
            if not transform.is_file():
                raise RuntimeError(f"Expected ANTs transform is missing: {transform}")
            shutil.copy2(transform, derivative_transform)
        _apply_background(output, mask_output, policy.background_fill)
        _validate_same_grid(output, mask_output)
        metadata = _read_sidecar(source)
        metadata.update(
            {
                "NeuroPrepExecutorVersion": "3.0",
                "SpatialReference": space,
                "SkullStripped": True,
                "BiasFieldCorrected": policy.n4_bias_correction,
                "Denoised": policy.denoise,
                "SingleInterpolationToReference": True,
                "TransformChain": [str(path) for path in transform_outputs],
                "QuantitativeValuesPreserved": policy.preserve_quantitative_values,
                "Units": "HU" if policy.modality == "ct" else metadata.get("Units"),
                "Sources": [str(source), str(fixed)],
                "GeneratedAt": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_json(_sidecar(output), metadata)
        _write_json(
            _sidecar(mask_output),
            {"Type": "Brain", "Sources": [str(source)], "SpatialReference": space},
        )
    return {
        "status": "pass" if execute else "planned",
        "input": str(source),
        "output": str(output),
        "mask": str(mask_output),
        "source_space": policy.modality,
        "target_space": space,
        "transforms": [str(path) for path in transform_outputs],
        "commands": commands,
        "single_resampling": True,
    }


def _discover_spatial_inputs(
    bids_root: Path, subject: str, modality: str, session: str | None
) -> list[Path]:
    suffix = STRUCTURAL_SUFFIXES[modality]
    subject_root = bids_root / f"sub-{subject}"
    search_root = subject_root / f"ses-{session.removeprefix('ses-')}" if session else subject_root
    patterns = [f"*_{suffix}.nii.gz"]
    if modality == "mra":
        patterns.extend(("*_MRA.nii.gz", "*_mra.nii.gz", "*_tof.nii.gz"))
    if modality == "ct":
        patterns.extend(("*_ct.nii.gz",))
    if modality == "swi":
        patterns.extend(
            (
                "*_SWI.nii.gz",
                "*_swi.nii.gz",
                "*_rec-SWI_*T2starw.nii.gz",
                "*_rec-minIP_*T2starw.nii.gz",
                "*_magnitude.nii.gz",
                "*_phase.nii.gz",
                "*_part-mag_*.nii.gz",
                "*_part-phase_*.nii.gz",
            )
        )
    discovered = sorted(
        {
            path
            for pattern in patterns
            for path in search_root.rglob(pattern)
            if "derivatives" not in path.parts and "desc-" not in path.name
        }
    )
    if modality == "swi":
        # A vendor-reconstructed SWI is immediately usable. Prefer it over
        # minIP and raw magnitude/phase components, which require separately
        # reviewed quantitative scaling before QSM reconstruction.
        reconstructed = [path for path in discovered if "_rec-swi_" in path.name.lower()]
        if reconstructed:
            return reconstructed
        minip = [path for path in discovered if "_rec-minip_" in path.name.lower()]
        if minip:
            return minip
    return discovered


def _discover_sensor_inputs(
    bids_root: Path, subject: str, modality: str, session: str | None
) -> list[Path]:
    subject_root = bids_root / f"sub-{subject}"
    search_root = subject_root / f"ses-{session.removeprefix('ses-')}" if session else subject_root
    extensions = (".fif", ".edf", ".bdf", ".vhdr", ".set", ".con", ".sqd", ".nwb")
    return sorted(
        path
        for path in search_root.rglob("*")
        if path.is_file()
        and path.parent.name == modality
        and path.name.lower().endswith(extensions)
        and "proc-" not in path.name
    )


def _resolve_reference(reference: Path | None, template: Path | None) -> Path | None:
    # An explicit template declares the requested final output grid. A visit-local
    # anatomical reference is only the fallback for native/subject-space workflows.
    for candidate in (template, reference):
        if candidate and candidate.is_file():
            return candidate
    for variable in ("NEUROPREP_STANDARD_TEMPLATE", "NEUROPREP_MNI_TEMPLATE"):
        environment = os.environ.get(variable)
        if environment and Path(environment).is_file():
            return Path(environment)
    try:
        from templateflow.api import get

        resolved = get(
            "MNI152NLin2009cAsym",
            resolution=2,
            suffix="T1w",
            desc=None,
            extension=".nii.gz",
        )
        candidates = [resolved] if isinstance(resolved, Path) else list(resolved or [])
        if len(candidates) == 1 and Path(candidates[0]).is_file():
            return Path(candidates[0])
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        pass
    return None


def _susceptibility_component(path: Path) -> str:
    name = path.name.lower()
    if "phase" in name or "part-phase" in name:
        return "phase"
    if "mag" in name or "magnitude" in name:
        return "magnitude"
    return "swi"


def _find_subject_reference(
    root: Path | None,
    subject: str,
    session: str | None = None,
) -> Path | None:
    if root is None or not root.exists():
        return None
    patterns = (
        f"sub-{subject}*_space-MNI152NLin2009cAsym*_desc-preproc_T1w.nii.gz",
        f"sub-{subject}*_space-MNI152NLin2009cAsym*_desc-noskull_T1w.nii.gz",
        f"sub-{subject}*_space-MNI152NLin2009cAsym*_desc-brain_T1w.nii.gz",
        f"sub-{subject}*_space-*_*T1w.nii.gz",
    )
    candidates = sorted({path for pattern in patterns for path in root.rglob(pattern)})
    if session:
        session_token = session.removeprefix("ses-")
        candidates = [
            path
            for path in candidates
            if f"_ses-{session_token}_" in path.name
            or f"ses-{session_token}" in path.parts
        ]
    return candidates[0] if candidates else None


def _session_entity(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("ses-"):
            return part
    match = re.search(r"(?:^|_)ses-([A-Za-z0-9]+)(?:_|$)", path.name)
    return f"ses-{match.group(1)}" if match else None


def _expected_transforms(prefix: Path, transform_type: str) -> list[Path]:
    affine = prefix.parent / f"{prefix.name}0GenericAffine.mat"
    warp = prefix.parent / f"{prefix.name}1Warp.nii.gz"
    return [warp, affine] if transform_type == "s" else [affine]


def _run(command: list[str]) -> None:
    if shutil.which(command[0]) is None:
        raise RuntimeError(
            f"Required executor tool is unavailable: {command[0]}. "
            "Run NeuroPrep inside the pinned production image or load the configured module."
        )
    subprocess.run(command, check=True)


def _apply_background(image_path: Path, mask_path: Path, fill: float) -> None:
    image = nib.load(str(image_path))
    mask_image = nib.load(str(mask_path))
    _validate_same_grid(image_path, mask_path)
    data = np.asarray(image.dataobj, dtype=np.float32)
    mask = np.asarray(mask_image.dataobj) > 0.5
    data[~mask] = np.float32(fill)
    nib.save(nib.Nifti1Image(data, image.affine, image.header), str(image_path))


def _validate_same_grid(first: Path, second: Path) -> None:
    first_image = nib.load(str(first))
    second_image = nib.load(str(second))
    if first_image.shape[:3] != second_image.shape[:3] or not np.allclose(
        first_image.affine, second_image.affine, atol=1e-5
    ):
        raise RuntimeError(f"Executor produced mismatched grids: {first} and {second}")


def _validate_nifti(path: Path, label: str) -> None:
    try:
        image = nib.load(str(path))
    except Exception as exc:
        raise ValueError(f"Unreadable {label}: {path}") from exc
    if len(image.shape) != 3:
        raise ValueError(f"{label.capitalize()} must be three-dimensional: {path}")


def _read_raw(mne: Any, path: Path) -> Any:
    name = path.name.lower()
    if name.endswith(".fif"):
        return mne.io.read_raw_fif(path, preload=False, verbose="error")
    if name.endswith(".edf"):
        return mne.io.read_raw_edf(path, preload=False, verbose="error")
    if name.endswith(".bdf"):
        return mne.io.read_raw_bdf(path, preload=False, verbose="error")
    if name.endswith(".vhdr"):
        return mne.io.read_raw_brainvision(path, preload=False, verbose="error")
    if name.endswith(".set"):
        return mne.io.read_raw_eeglab(path, preload=False, verbose="error")
    if name.endswith((".con", ".sqd")):
        return mne.io.read_raw_kit(path, preload=False, verbose="error")
    if name.endswith(".nwb") and hasattr(mne.io, "read_raw_nwb"):
        return mne.io.read_raw_nwb(path, preload=False, verbose="error")
    raise ValueError(f"Unsupported sensor recording format: {path}")


def _detect_bad_channels(raw: Any) -> list[str]:
    data = raw.get_data(reject_by_annotation="omit")
    if data.size == 0:
        raise ValueError("Sensor recording contains no usable samples")
    variances = np.nanvar(data, axis=1)
    positive = variances[np.isfinite(variances) & (variances > 0)]
    if positive.size == 0:
        return list(raw.ch_names)
    median = float(np.median(positive))
    mad = float(np.median(np.abs(positive - median))) or median * 0.01
    low = max(np.finfo(float).tiny, median - 8 * mad)
    high = median + 8 * mad
    return [
        name
        for name, variance in zip(raw.ch_names, variances, strict=True)
        if not np.isfinite(variance) or variance < low or variance > high
    ]


def _apply_ica(mne: Any, raw: Any) -> list[int]:
    picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=False, ecg=False, exclude="bads")
    if len(picks) < 4:
        return []
    ica = mne.preprocessing.ICA(n_components=0.99, random_state=42, method="fastica")
    ica.fit(raw, picks=picks, decim=max(1, int(raw.info["sfreq"] // 100)), verbose="error")
    excluded: set[int] = set()
    if mne.pick_types(raw.info, eog=True).size:
        indices, _ = ica.find_bads_eog(raw, verbose="error")
        excluded.update(indices)
    if mne.pick_types(raw.info, ecg=True).size:
        indices, _ = ica.find_bads_ecg(raw, verbose="error")
        excluded.update(indices)
    ica.exclude = sorted(excluded)
    if ica.exclude:
        ica.apply(raw, verbose="error")
    return list(ica.exclude)


def _copy_sensor_companions(source: Path, destination: Path) -> dict[str, str]:
    prefix = _bids_prefix(source)
    copied: dict[str, str] = {}
    for label, suffix in (
        ("channels", "channels.tsv"),
        ("events", "events.tsv"),
        ("electrodes", "electrodes.tsv"),
        ("coordsystem", "coordsystem.json"),
    ):
        candidate = source.with_name(f"{prefix}_{suffix}")
        if candidate.exists():
            target = destination / candidate.name
            shutil.copy2(candidate, target)
            copied[label] = str(target)
    return copied


def _write_sensor_report(path: Path, record: dict[str, Any]) -> None:
    escaped = (
        json.dumps(record, indent=2, sort_keys=True).replace("&", "&amp;").replace("<", "&lt;")
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>NeuroPrep sensor QC</title>"
        "</head><body><h1>NeuroPrep sensor preprocessing report</h1><pre>"
        + escaped
        + "</pre></body></html>\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _relative_subject_directory(source: Path, subject: str, datatype: str) -> Path:
    session = next((part for part in source.parts if part.startswith("ses-")), None)
    return Path(f"sub-{subject}") / (session or "") / datatype


def _bids_prefix(path: Path) -> str:
    stem = path.name
    for ending in (
        ".nii.gz",
        ".nii",
        ".fif",
        ".edf",
        ".bdf",
        ".vhdr",
        ".set",
        ".con",
        ".sqd",
        ".nwb",
    ):
        if stem.lower().endswith(ending):
            stem = stem[: -len(ending)]
            break
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def _space_label(reference: Path) -> str:
    for token in reference.name.split("_"):
        if token.startswith("space-"):
            return token.removeprefix("space-")
        if token.startswith("tpl-"):
            return token.removeprefix("tpl-")
    return "MNI152NLin2009cAsym"


def _sidecar(path: Path) -> Path:
    name = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    return path.with_name(name + ".json")


def _read_sidecar(path: Path) -> dict[str, Any]:
    sidecar = _sidecar(path)
    if not sidecar.exists():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
