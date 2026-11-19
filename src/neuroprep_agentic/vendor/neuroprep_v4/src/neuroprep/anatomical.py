from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .visual_qc import write_image_qc_report


SPACE_PATTERN = re.compile(r"(?:^|_)space-([^_]+)")
PREPROC_SUFFIX = "_desc-preproc_T1w.nii.gz"
DEEPPREP_BRAIN_SUFFIX = "_desc-noskull_T1w.nii.gz"
MASK_SUFFIX = "_desc-brain_mask.nii.gz"
FINAL_SUFFIX = "_desc-brain_T1w.nii.gz"
STANDARD_SPACE = "MNI152NLin2009cAsym"
REGISTRATION_CONTAINER_ENV = "NEUROPREP_NATIVE_CONTAINER"
STANDARD_TEMPLATE_ENV = "NEUROPREP_STANDARD_TEMPLATE"
STANDARD_MASK_ENV = "NEUROPREP_STANDARD_MASK"
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


def finalize_anatomical_subject(
    preprocessing_root: Path,
    output_root: Path,
    subject: str,
    marker: Path,
    required_spaces: tuple[str, ...] = (),
    source_pipeline: str = "smriprep",
    registration_container: Path | None = None,
    standard_template: Path | None = None,
    standard_mask: Path | None = None,
    threads: int = 8,
) -> dict[str, Any]:
    """Create validated skull-stripped derivatives from a reviewed T1w executor."""

    subject_token = subject.removeprefix("sub-")
    subject_root = _find_subject_root(preprocessing_root, subject_token)
    preparation_records: list[dict[str, Any]] = []
    if source_pipeline == "deepprep":
        preparation_records = _prepare_deepprep_anatomy(
            subject_root=subject_root,
            subject=subject_token,
            required_spaces=required_spaces,
            registration_container=_path_or_environment(
                registration_container, REGISTRATION_CONTAINER_ENV
            ),
            standard_template=_path_or_environment(
                standard_template, STANDARD_TEMPLATE_ENV
            ),
            standard_mask=_path_or_environment(standard_mask, STANDARD_MASK_ENV),
            threads=threads,
        )
    preprocessed = sorted(subject_root.rglob(f"*{PREPROC_SUFFIX}"))
    if source_pipeline == "deepprep":
        preprocessed = sorted(
            set(preprocessed) | set(subject_root.rglob(f"*{DEEPPREP_BRAIN_SUFFIX}"))
        )
    if not preprocessed:
        raise FileNotFoundError(f"No preprocessed T1w images found under {subject_root}")
    reports = _find_reports(preprocessing_root, subject_root, subject_token)
    if not reports:
        raise FileNotFoundError(
            f"No {source_pipeline} visual QC report for sub-{subject_token} under "
            f"{preprocessing_root}"
        )

    records: list[dict[str, Any]] = []
    observed_spaces: set[str] = set()
    failures: list[str] = []
    for preproc_path in preprocessed:
        if preproc_path.name.endswith(DEEPPREP_BRAIN_SUFFIX):
            source_stem = preproc_path.name.removesuffix(DEEPPREP_BRAIN_SUFFIX)
            relative = preproc_path.relative_to(preprocessing_root)
            mask_path = output_root / relative.parent / (source_stem + MASK_SUFFIX)
            _write_nonzero_mask(preproc_path, mask_path)
            output_name = source_stem + FINAL_SUFFIX
        else:
            source_stem = preproc_path.name.removesuffix(PREPROC_SUFFIX)
            mask_path = preproc_path.with_name(source_stem + MASK_SUFFIX)
            relative = preproc_path.relative_to(preprocessing_root)
            output_name = source_stem + FINAL_SUFFIX
        if not mask_path.exists():
            failures.append(f"missing-mask:{mask_path}")
            continue
        output_path = output_root / relative.parent / output_name
        record = _mask_image(
            preproc_path,
            mask_path,
            output_path,
            source_pipeline=source_pipeline,
        )
        report_path = output_path.with_name(
            output_path.name.removesuffix(".nii.gz") + "_qc.html"
        )
        write_image_qc_report(
            output_path,
            mask_path,
            report_path,
            "NeuroPrep T1w model-ready QC",
        )
        record["visual_reports"] = [str(report_path), *[str(path) for path in reports]]
        records.append(record)
        observed_spaces.add(str(record["space"]))
        failures.extend(
            f"{output_path.name}:{flag}" for flag in record["qc"]["flags"]
        )

    missing_spaces = sorted(set(required_spaces) - observed_spaces)
    failures.extend(f"missing-required-space:{space}" for space in missing_spaces)
    status = "pass" if records and not failures else "fail"
    summary = {
        "status": status,
        "subject": subject_token,
        "modality": "t1w",
        "pipeline": source_pipeline,
        "preprocessing_root": str(preprocessing_root),
        "output_root": str(output_root),
        "required_spaces": list(required_spaces),
        "observed_spaces": sorted(observed_spaces),
        "visual_reports": [str(path) for path in reports],
        "corrections": _corrections(source_pipeline),
        "preparation_records": preparation_records,
        "records": records,
        "failures": failures,
    }
    _write_json(marker, summary)
    return summary


def _prepare_deepprep_anatomy(
    subject_root: Path,
    subject: str,
    required_spaces: tuple[str, ...],
    registration_container: Path | None,
    standard_template: Path | None,
    standard_mask: Path | None,
    threads: int,
) -> list[dict[str, Any]]:
    """Convert visit-specific DeepPrep MGZ brains and normalize them when required."""

    if threads < 1:
        raise ValueError("threads must be at least 1")
    brain_sources = _find_deepprep_brains(subject_root, subject)
    if not brain_sources:
        return []

    require_standard = STANDARD_SPACE in required_spaces
    if require_standard:
        missing = [
            label
            for label, path in (
                ("registration container", registration_container),
                ("standard template", standard_template),
                ("standard mask", standard_mask),
            )
            if path is None or not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "DeepPrep standard-space preparation requires: " + ", ".join(missing)
            )
        assert registration_container is not None
        assert standard_template is not None
        assert standard_mask is not None
        _validate_template_pair(standard_template, standard_mask)

    records: list[dict[str, Any]] = []
    for source in brain_sources:
        session, preparation_root = _deepprep_visit_location(
            subject_root, source, subject
        )
        entity_prefix = f"sub-{subject}"
        if session:
            entity_prefix += f"_ses-{session}"
        native_stem = f"{entity_prefix}_space-subject-anat"
        native = preparation_root / f"{native_stem}{DEEPPREP_BRAIN_SUFFIX}"
        native_mask = preparation_root / f"{native_stem}{MASK_SUFFIX}"
        if not _valid_spatial_image(native):
            _convert_mgz_brain(source, native)
        if not _valid_spatial_image(native_mask, reference=native):
            _write_nonzero_mask(native, native_mask)
        _write_json(
            _sidecar(native),
            {
                "NeuroPrepContractVersion": "3.0",
                "SkullStripped": True,
                "SpatialReference": "subject-anat",
                "SpatiallyNormalized": False,
                "Sources": [str(source)],
                "GeneratedBy": [
                    {
                        "Name": "NeuroPrep DeepPrep anatomical adapter",
                        "Description": (
                            "Lossless MGZ-to-NIfTI conversion of DeepPrep/FastSurfer "
                            "brain.mgz without spatial resampling"
                        ),
                    }
                ],
                "CorrectionsApplied": _corrections("deepprep"),
            },
        )
        record: dict[str, Any] = {
            "session": session,
            "source": str(source),
            "native": str(native),
            "native_mask": str(native_mask),
            "standard": None,
            "standard_mask": None,
            "registration_reused": None,
        }

        if require_standard:
            assert registration_container is not None
            assert standard_template is not None
            assert standard_mask is not None
            standard_stem = (
                f"{entity_prefix}_space-{STANDARD_SPACE}_res-02"
            )
            standard = preparation_root / f"{standard_stem}{DEEPPREP_BRAIN_SUFFIX}"
            normalized_mask = preparation_root / f"{standard_stem}{MASK_SUFFIX}"
            registration_reused = _valid_spatial_image(
                standard, reference=standard_template
            )
            if not registration_reused:
                _register_to_standard(
                    native=native,
                    native_mask=native_mask,
                    output=standard,
                    container=registration_container,
                    template=standard_template,
                    template_mask=standard_mask,
                    threads=threads,
                )
            if not _valid_spatial_image(normalized_mask, reference=standard_template):
                _copy_binary_mask(standard_mask, normalized_mask)
            _apply_mask_in_place(standard, normalized_mask)
            registration_root = preparation_root / ".registration"
            transforms = [
                registration_root / "from-subject_to-MNI_0GenericAffine.mat",
                registration_root / "from-subject_to-MNI_1Warp.nii.gz",
            ]
            _write_json(
                _sidecar(standard),
                {
                    "NeuroPrepContractVersion": "3.0",
                    "SkullStripped": True,
                    "SpatialReference": STANDARD_SPACE,
                    "SpatiallyNormalized": True,
                    "Resolution": "02",
                    "Sources": [str(native), str(source)],
                    "Template": str(standard_template),
                    "TemplateSHA256": _sha256(standard_template),
                    "TemplateMask": str(standard_mask),
                    "Transforms": [str(path) for path in transforms],
                    "GeneratedBy": [
                        {
                            "Name": "NeuroPrep DeepPrep anatomical adapter",
                            "Description": (
                                "ANTs SyN registration of the visit-specific "
                                "skull-stripped T1w image to MNI152NLin2009cAsym"
                            ),
                        }
                    ],
                    "CorrectionsApplied": _corrections("deepprep"),
                },
            )
            record.update(
                {
                    "standard": str(standard),
                    "standard_mask": str(normalized_mask),
                    "registration_reused": registration_reused,
                }
            )
        records.append(record)
    return records


def _find_deepprep_brains(subject_root: Path, subject: str) -> list[Path]:
    unique: dict[Path, Path] = {}
    for path in subject_root.rglob(f"sub-{subject}/mri/brain.mgz"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        unique.setdefault(resolved, path)
    return sorted(unique.values())


def _deepprep_visit_location(
    subject_root: Path, source: Path, subject: str
) -> tuple[str | None, Path]:
    relative = source.relative_to(subject_root)
    parts = relative.parts
    session: str | None = None
    visit_root = subject_root
    if "visits" in parts:
        visit_index = parts.index("visits")
        if len(parts) > visit_index + 1:
            visit_name = parts[visit_index + 1]
            session = visit_name.removeprefix("ses-")
            visit_root = subject_root.joinpath(*parts[: visit_index + 2])
    if session is None:
        session = next(
            (
                part.removeprefix("ses-")
                for part in parts
                if part.startswith("ses-")
            ),
            None,
        )
    return session, visit_root / "AnatPrep" / f"sub-{subject}" / "anat"


def _convert_mgz_brain(source: Path, output: Path) -> None:
    image = nib.as_closest_canonical(nib.load(str(source)))
    data = np.asanyarray(image.dataobj, dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"DeepPrep brain image must be 3D: {source} has {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"DeepPrep brain image contains non-finite values: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    converted = nib.Nifti1Image(data, image.affine)
    converted.header.set_data_dtype(np.float32)
    converted.set_qform(image.affine, code=1)
    converted.set_sform(image.affine, code=1)
    _atomic_nifti(converted, output)


def _register_to_standard(
    native: Path,
    native_mask: Path,
    output: Path,
    container: Path,
    template: Path,
    template_mask: Path,
    threads: int,
) -> None:
    runtime = shutil.which("apptainer") or shutil.which("singularity")
    if runtime is None:
        raise FileNotFoundError("Neither apptainer nor singularity is available")
    registration_root = output.parent / ".registration"
    registration_root.mkdir(parents=True, exist_ok=True)
    prefix = registration_root / "from-subject_to-MNI_"
    bindings = [
        f"{output.parent}:/work",
        f"{template.parent}:/template:ro",
        f"{template_mask.parent}:/template-mask:ro",
    ]
    common = [
        runtime,
        "exec",
        "--cleanenv",
        "--env",
        f"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={threads}",
    ]
    for binding in bindings:
        common.extend(["--bind", binding])
    common.append(str(container))
    register = [
        *common,
        "/app/.pixi/envs/qsiprep/bin/antsRegistrationSyNQuick.sh",
        "-d",
        "3",
        "-f",
        f"/template/{template.name}",
        "-m",
        f"/work/{native.name}",
        "-o",
        "/work/.registration/from-subject_to-MNI_",
        "-n",
        str(threads),
        "-t",
        "s",
        "-x",
        (
            f"/template-mask/{template_mask.name},"
            f"/work/{native_mask.name}"
        ),
        "-p",
        "f",
        "-y",
        "1",
    ]
    _run_logged(register, registration_root / "registration.log")
    affine = prefix.with_name(prefix.name + "0GenericAffine.mat")
    warp = prefix.with_name(prefix.name + "1Warp.nii.gz")
    for transform in (affine, warp):
        if not transform.is_file():
            raise FileNotFoundError(f"Registration did not create {transform}")
    apply = [
        *common,
        "/app/.pixi/envs/qsiprep/bin/antsApplyTransforms",
        "-d",
        "3",
        "-i",
        f"/work/{native.name}",
        "-r",
        f"/template/{template.name}",
        "-o",
        f"/work/{output.name}",
        "-n",
        "BSpline",
        "-t",
        "/work/.registration/from-subject_to-MNI_1Warp.nii.gz",
        "-t",
        "/work/.registration/from-subject_to-MNI_0GenericAffine.mat",
    ]
    _run_logged(apply, registration_root / "apply-transforms.log")
    if not _valid_spatial_image(output, reference=template):
        raise ValueError(f"Invalid standard-space anatomical output: {output}")


def _run_logged(command: list[str], log_path: Path) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    log_path.write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        tail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(
            f"Command failed with status {completed.returncode}: "
            f"{' '.join(command[:4])}\n{tail}"
        )


def _copy_binary_mask(source: Path, output: Path) -> None:
    image = nib.load(str(source))
    data = np.asanyarray(image.dataobj)
    mask = np.isfinite(data) & (data > 0)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    _atomic_nifti(
        nib.Nifti1Image(mask.astype(np.uint8), image.affine, header),
        output,
    )


def _apply_mask_in_place(image_path: Path, mask_path: Path) -> None:
    image = nib.load(str(image_path))
    mask_image = nib.load(str(mask_path))
    if image.shape != mask_image.shape or not np.allclose(
        image.affine, mask_image.affine, atol=1e-5
    ):
        raise ValueError(f"Standard image/mask geometry mismatch: {image_path}")
    data = np.asanyarray(image.dataobj, dtype=np.float32)
    mask = np.asanyarray(mask_image.dataobj) > 0
    data[~mask] = 0
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    _atomic_nifti(nib.Nifti1Image(data, image.affine, header), image_path)


def _valid_spatial_image(path: Path, reference: Path | None = None) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        image = nib.load(str(path))
        if len(image.shape) != 3 or min(image.shape) < 2:
            return False
        if reference is not None:
            reference_image = nib.load(str(reference))
            return image.shape == reference_image.shape and np.allclose(
                image.affine, reference_image.affine, atol=1e-4
            )
        return True
    except (OSError, ValueError):
        return False


def _validate_template_pair(template: Path, mask: Path) -> None:
    if not _valid_spatial_image(template):
        raise ValueError(f"Invalid standard template: {template}")
    if not _valid_spatial_image(mask, reference=template):
        raise ValueError(f"Standard template/mask geometry mismatch: {mask}")


def _path_or_environment(path: Path | None, variable: str) -> Path | None:
    if path is not None:
        return path
    value = os.environ.get(variable)
    return Path(value) if value else None


def _mask_image(
    preproc_path: Path,
    mask_path: Path,
    output_path: Path,
    source_pipeline: str = "smriprep",
) -> dict[str, Any]:
    preproc_image = nib.load(str(preproc_path))
    mask_image = nib.load(str(mask_path))
    if preproc_image.shape != mask_image.shape:
        raise ValueError(
            f"Image/mask shape mismatch: {preproc_path} {preproc_image.shape} != "
            f"{mask_path} {mask_image.shape}"
        )
    if not np.allclose(preproc_image.affine, mask_image.affine, atol=1e-5):
        raise ValueError(f"Image/mask affine mismatch: {preproc_path} != {mask_path}")

    data = np.asanyarray(preproc_image.dataobj, dtype=np.float32)
    mask_values = np.asanyarray(mask_image.dataobj, dtype=np.float32)
    mask = np.isfinite(mask_values) & (mask_values > 0.5)
    flags: list[str] = []
    if not mask.any():
        flags.append("empty-brain-mask")
    if mask.all():
        flags.append("brain-mask-covers-entire-grid")
    if not np.isfinite(data[mask]).all():
        flags.append("non-finite-voxels-inside-brain")

    masked = np.zeros(data.shape, dtype=np.float32)
    masked[mask] = data[mask]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = preproc_image.header.copy()
    header.set_data_dtype(np.float32)
    output_image = nib.Nifti1Image(masked, preproc_image.affine, header)
    output_image.set_qform(
        preproc_image.get_qform(), int(preproc_image.header["qform_code"])
    )
    output_image.set_sform(
        preproc_image.get_sform(), int(preproc_image.header["sform_code"])
    )
    _atomic_nifti(output_image, output_path)

    outside_nonzero = int(np.count_nonzero(masked[~mask]))
    inside = masked[mask]
    finite_fraction = float(np.isfinite(masked).mean())
    nonzero_inside_fraction = float(np.count_nonzero(inside) / inside.size) if inside.size else 0.0
    if outside_nonzero:
        flags.append("nonzero-voxels-outside-brain-mask")
    if finite_fraction < 1.0:
        flags.append("non-finite-output-voxels")
    if nonzero_inside_fraction <= 0.01:
        flags.append("nearly-empty-brain-image")

    space_match = SPACE_PATTERN.search(preproc_path.name)
    space = space_match.group(1) if space_match else "subject-anat"
    metadata_path = _sidecar(output_path)
    source_metadata_path = _sidecar(preproc_path)
    source_metadata: dict[str, Any] = {}
    if source_metadata_path.exists():
        try:
            source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            flags.append("invalid-source-sidecar")
    metadata = {
        **source_metadata,
        "NeuroPrepContractVersion": "3.0",
        "ModelReady": not flags,
        "SkullStripped": True,
        "SpatialReference": space,
        "SpatiallyNormalized": space != "subject-anat",
        "Sources": [str(preproc_path), str(mask_path)],
        "GeneratedBy": [
            {
                "Name": "NeuroPrep anatomical finalizer",
                "Description": (
                    f"Applied the {source_pipeline} brain mask without additional resampling"
                ),
            }
        ],
        "CorrectionsApplied": _corrections(source_pipeline),
    }
    _write_json(metadata_path, metadata)
    record = {
        "modality": "t1w",
        "payload_kind": "spatial-image",
        "space": space,
        "units": "arbitrary",
        "site_covariates": {
            field: source_metadata[field]
            for field in SITE_METADATA_FIELDS
            if field in source_metadata
        },
        "input": str(preproc_path),
        "mask": str(mask_path),
        "output": str(output_path),
        "metadata": str(metadata_path),
        "auxiliaries": {},
        "sha256": _sha256(output_path),
        "qc": {
            "status": "pass" if not flags else "fail",
            "flags": flags,
            "shape": list(masked.shape),
            "voxel_sizes_mm": [
                float(value) for value in output_image.header.get_zooms()[:3]
            ],
            "finite_fraction": finite_fraction,
            "brain_mask_fraction": float(mask.mean()),
            "nonzero_inside_fraction": nonzero_inside_fraction,
            "nonzero_outside_voxels": outside_nonzero,
        },
    }
    registration_reference = source_metadata.get("Template")
    registration_reference_mask = source_metadata.get("TemplateMask")
    if (
        space != "subject-anat"
        and registration_reference
        and registration_reference_mask
    ):
        record["registration_reference"] = str(registration_reference)
        record["registration_reference_mask"] = str(registration_reference_mask)
    return record


def _find_subject_root(root: Path, subject: str) -> Path:
    direct = root / f"sub-{subject}"
    if direct.is_dir():
        return direct
    candidates = sorted(path for path in root.rglob(f"sub-{subject}") if path.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    if not root.exists():
        raise FileNotFoundError(f"Anatomical derivative root does not exist: {root}")
    raise FileNotFoundError(f"No unique sub-{subject} directory under {root}")


def _find_reports(root: Path, subject_root: Path, subject: str) -> list[Path]:
    reports = set(root.glob(f"sub-{subject}*.html"))
    reports.update(root.rglob(f"*sub-{subject}*.html"))
    reports.update(subject_root.rglob("*.html"))
    if not reports:
        reports.update(root.glob("QC/*.html"))
        reports.update(root.glob("QC/**/*.html"))
    return sorted(path for path in reports if path.is_file())


def _write_nonzero_mask(image_path: Path, mask_path: Path) -> None:
    """Materialize the mask implicit in DeepPrep's desc-noskull derivative."""

    image = nib.load(str(image_path))
    data = np.asanyarray(image.dataobj)
    mask = np.isfinite(data) & (data != 0)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    _atomic_nifti(
        nib.Nifti1Image(mask.astype(np.uint8), image.affine, header),
        mask_path,
    )


def _corrections(source_pipeline: str) -> list[str]:
    if source_pipeline == "deepprep":
        return [
            "within-participant T1w motion correction when multiple scans are present",
            "N4 bias-field correction",
            "FastSurferCNN skull stripping and tissue segmentation",
            "ANTs SyN spatial normalization for standard-space derivatives",
            "brain-mask application",
        ]
    return [
        "ANTs DenoiseImage",
        "N4 bias-field correction",
        "intensity normalization",
        "brain-mask application",
    ]


def _sidecar(path: Path) -> Path:
    return path.with_suffix("").with_suffix(".json")


def _atomic_nifti(image: nib.Nifti1Image, path: Path) -> None:
    temporary = path.parent / f".{path.name}.tmp.nii.gz"
    nib.save(image, str(temporary))
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
