from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


SENSOR_MODALITIES = {"eeg", "meg", "ieeg"}


@dataclass(frozen=True)
class GridSpec:
    shape: tuple[int, int, int]
    affine: tuple[tuple[float, ...], ...]
    orientation: tuple[str, str, str]
    voxel_sizes_mm: tuple[float, float, float]
    grid_id: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        payload["affine"] = [list(row) for row in self.affine]
        payload["orientation"] = list(self.orientation)
        payload["voxel_sizes_mm"] = list(self.voxel_sizes_mm)
        return payload


def grid_spec(path: Path) -> GridSpec:
    image = nib.load(str(path))
    if len(image.shape) < 3:
        raise ValueError(f"Grid reference must have at least three dimensions: {path}")
    shape = tuple(int(value) for value in image.shape[:3])
    affine_array = np.asarray(image.affine, dtype=float)
    identity = {
        "shape": shape,
        "affine": np.round(affine_array, 8).tolist(),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return GridSpec(
        shape=shape,
        affine=tuple(tuple(float(value) for value in row) for row in affine_array),
        orientation=tuple(str(value) for value in nib.aff2axcodes(affine_array)),
        voxel_sizes_mm=tuple(float(value) for value in image.header.get_zooms()[:3]),
        grid_id=f"grid-{digest}",
    )


def create_common_grid_reference(
    sample_index: Path,
    output: Path,
    preferred_modalities: tuple[str, ...] = ("t1w", "t2w", "flair", "ct"),
    required_space: str = "MNI152NLin2009cAsym",
) -> dict[str, Any]:
    rows = _read_jsonl(sample_index)
    candidates = [
        row
        for modality in preferred_modalities
        for row in rows
        if row.get("modality") == modality
        and required_space.lower() in str(row.get("space", "")).lower()
        and Path(str(row.get("path", ""))).exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No passing {required_space} image found in {sample_index} for grid construction"
        )
    candidates.sort(
        key=lambda row: (
            preferred_modalities.index(str(row["modality"])),
            str(row.get("participant_id", "")),
            str(row.get("path", "")),
        )
    )
    source = Path(str(candidates[0]["path"]))
    image = nib.load(str(source))
    reference = nib.Nifti1Image(
        np.zeros(image.shape[:3], dtype=np.uint8), image.affine, image.header
    )
    reference.set_data_dtype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.nii.gz")
    nib.save(reference, str(temporary))
    temporary.replace(output)
    report = {
        "status": "pass",
        "source": str(source),
        "source_modality": candidates[0]["modality"],
        "required_space": required_space,
        "reference": str(output),
        "grid": grid_spec(output).to_dict(),
        "selection_policy": "preferred modality then participant and path lexical order",
    }
    _write_json(_sidecar(output), report)
    return report


def validate_common_grid(
    paths: list[Path], reference: Path, affine_tolerance: float = 1e-5
) -> dict[str, Any]:
    target = grid_spec(reference)
    records: list[dict[str, Any]] = []
    for path in paths:
        current = grid_spec(path)
        shape_match = current.shape == target.shape
        affine_match = np.allclose(current.affine, target.affine, atol=affine_tolerance, rtol=0)
        orientation_match = current.orientation == target.orientation
        records.append(
            {
                "path": str(path),
                "status": "pass" if shape_match and affine_match and orientation_match else "fail",
                "shape_match": shape_match,
                "affine_match": bool(affine_match),
                "orientation_match": orientation_match,
                "grid_id": current.grid_id,
            }
        )
    return {
        "status": "pass" if records and all(row["status"] == "pass" for row in records) else "fail",
        "reference": str(reference),
        "target_grid": target.to_dict(),
        "records": records,
    }


def align_visit(
    visit_manifest: Path,
    reference: Path | None,
    output_root: Path,
    sensor_transform_root: Path | None = None,
    require_standard_sensor_coordinates: bool = False,
) -> dict[str, Any]:
    """Align one visit to one explicit grid without resampling raw DWI gradients."""

    visit = _read_object(visit_manifest)
    target_image = nib.load(str(reference)) if reference is not None else None
    target = grid_spec(reference) if reference is not None else None
    participant = str(visit["participant_id"])
    visit_id = str(visit["visit_id"])
    visit_output = output_root / f"sub-{participant}" / visit_id
    aligned: dict[str, Any] = {}
    failures: list[str] = []

    for modality, entry in sorted(visit.get("modalities", {}).items()):
        if not isinstance(entry, dict) or not entry.get("present"):
            aligned[modality] = {"present": False, "missing": True}
            continue
        sample = entry.get("primary")
        if not isinstance(sample, dict):
            failures.append(f"missing-primary:{modality}")
            continue
        if modality in SENSOR_MODALITIES:
            try:
                aligned[modality] = _align_sensor(
                    sample,
                    visit_output / modality,
                    sensor_transform_root,
                    participant,
                    visit_id,
                    require_standard_sensor_coordinates,
                )
            except (FileNotFoundError, ValueError) as exc:
                failures.append(f"sensor-alignment:{modality}:{exc}")
            continue
        if modality == "dwi" and sample.get("payload_kind") == "gradient-image":
            try:
                if target_image is None:
                    raise ValueError("DWI shell maps require an explicit voxel-grid reference")
                aligned[modality] = _dwi_shell_maps(sample, target_image, visit_output / modality)
            except (FileNotFoundError, ValueError) as exc:
                failures.append(f"dwi-shell-map:{exc}")
            continue
        try:
            if target_image is None or target is None:
                raise ValueError(
                    f"Spatial modality {modality} requires an explicit voxel-grid reference"
                )
            aligned[modality] = _align_image_sample(
                sample, target_image, target, visit_output / modality
            )
        except (FileNotFoundError, ValueError) as exc:
            failures.append(f"image-alignment:{modality}:{exc}")

    grid_paths: list[Path] = []
    if target is not None:
        for record in aligned.values():
            if not isinstance(record, dict):
                continue
            if record.get("grid_id") == target.grid_id and record.get("path"):
                grid_paths.append(Path(str(record["path"])))
            for channel in record.get("shell_channels", []):
                if (
                    isinstance(channel, dict)
                    and channel.get("grid_id") == target.grid_id
                    and channel.get("path")
                ):
                    grid_paths.append(Path(str(channel["path"])))
    validation = (
        validate_common_grid(grid_paths, reference)
        if grid_paths and reference is not None
        else {"status": "not-applicable", "records": [], "reason": "sensor-only visit"}
    )
    has_spatial_output = any(
        isinstance(record, dict) and (record.get("stackable") or record.get("shell_channels"))
        for record in aligned.values()
    )
    if has_spatial_output and validation["status"] != "pass":
        failures.append("common-grid-validation-failed")
    result = {
        "contract_version": "3.0",
        "dataset_id": visit.get("dataset_id"),
        "participant_id": participant,
        "visit_id": visit_id,
        "visit_key": visit.get("visit_key"),
        "days_from_baseline": visit.get("days_from_baseline"),
        "site_covariates": visit.get("site_covariates", {}),
        "site_covariate_conflicts": visit.get("site_covariate_conflicts", []),
        "target_grid": target.to_dict() if target is not None else None,
        "reference": str(reference) if reference is not None else None,
        "single_resampling_policy": True,
        "modalities": aligned,
        "missing_modality_mask": visit.get("missing_modality_mask", {}),
        "grid_validation": validation,
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    output = visit_output / f"sub-{participant}_{visit_id}_aligned-visit.json"
    _write_json(output, result)
    result["manifest_path"] = str(output)
    return result


def align_dataset(
    visit_index: Path,
    reference: Path | None,
    output_root: Path,
    output_index: Path,
    sensor_transform_root: Path | None = None,
    require_standard_sensor_coordinates: bool = False,
) -> dict[str, Any]:
    visits = _read_jsonl(visit_index)
    results = []
    failures = []
    excluded = []
    for visit in visits:
        if visit.get("status") != "pass":
            excluded.append(
                {
                    "visit_key": visit.get("visit_key"),
                    "participant_id": visit.get("participant_id"),
                    "visit_id": visit.get("visit_id"),
                    "status": visit.get("status"),
                    "reason": visit.get("exclusion_reason"),
                    "raw_acquired_modalities": visit.get("raw_acquired_modalities", []),
                    "processing_missing_modalities": visit.get(
                        "processing_missing_modalities", []
                    ),
                }
            )
            continue
        manifest_value = visit.get("manifest_path")
        if not manifest_value:
            failures.append(f"missing-manifest-path:{visit.get('visit_key')}")
            continue
        try:
            result = align_visit(
                Path(str(manifest_value)),
                reference,
                output_root,
                sensor_transform_root,
                require_standard_sensor_coordinates,
            )
        except (FileNotFoundError, ValueError) as exc:
            failures.append(f"{visit.get('visit_key')}:{exc}")
            continue
        results.append(result)
        if result["status"] != "pass":
            failures.append(f"{visit.get('visit_key')}:{';'.join(result['failures'])}")
    output_index.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_index.with_suffix(output_index.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
    temporary.replace(output_index)
    summary = {
        "status": "pass" if results and not failures else "fail",
        "visits": len(visits),
        "aligned": len(results),
        "excluded": len(excluded),
        "excluded_visits": excluded,
        "failures": failures,
        "index": str(output_index),
        "reference": str(reference) if reference is not None else None,
    }
    _write_json(output_index.with_suffix(output_index.suffix + ".summary.json"), summary)
    return summary


def _align_image_sample(
    sample: dict[str, Any], target_image: Any, target: GridSpec, output_dir: Path
) -> dict[str, Any]:
    source_path = Path(str(sample["path"]))
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source = nib.load(str(source_path))
    if len(source.shape) not in {3, 4}:
        raise ValueError(f"unsupported image dimensionality {source.shape}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / _aligned_name(source_path.name, target.grid_id)
    already_aligned = source.shape[:3] == target.shape and np.allclose(
        source.affine, target_image.affine, atol=1e-5, rtol=0
    )
    if already_aligned:
        _atomic_copy(source_path, output)
        resampled = False
    else:
        order = _interpolation_order(str(sample.get("modality", "")))
        _resample_image(source, target_image, output, order)
        resampled = True
    mask_output = None
    if sample.get("mask"):
        mask_path = Path(str(sample["mask"]))
        if mask_path.exists():
            mask_output = output_dir / _aligned_name(mask_path.name, target.grid_id)
            mask = nib.load(str(mask_path))
            if mask.shape[:3] == target.shape and np.allclose(
                mask.affine, target_image.affine, atol=1e-5, rtol=0
            ):
                _atomic_copy(mask_path, mask_output)
            else:
                _resample_image(mask, target_image, mask_output, 0)
            _apply_mask(output, mask_output, str(sample.get("modality", "")))
    metadata = {
        "NeuroPrepContractVersion": "3.0",
        "Sources": [str(source_path), str(target_image.get_filename() or "explicit-reference")],
        "GridID": target.grid_id,
        "ResampledInAlignmentStage": resampled,
        "InterpolationOrder": _interpolation_order(str(sample.get("modality", ""))),
        "SingleResamplingPolicy": True,
        "OriginalMetadata": sample.get("metadata"),
    }
    _write_json(_sidecar(output), metadata)
    return {
        **{key: value for key, value in sample.items() if key not in {"path", "mask"}},
        "present": True,
        "path": str(output),
        "mask": str(mask_output) if mask_output else None,
        "grid_id": target.grid_id,
        "resampled": resampled,
        "stackable": True,
    }


def _dwi_shell_maps(sample: dict[str, Any], target_image: Any, output_dir: Path) -> dict[str, Any]:
    source_path = Path(str(sample["path"]))
    auxiliaries = sample.get("auxiliaries") if isinstance(sample.get("auxiliaries"), dict) else {}
    bval_path = Path(str(auxiliaries.get("bval", "")))
    bvec_path = Path(str(auxiliaries.get("bvec", "")))
    if not source_path.exists() or not bval_path.exists() or not bvec_path.exists():
        raise FileNotFoundError("DWI, bval, and bvec are required")
    image = nib.load(str(source_path))
    if len(image.shape) != 4:
        raise ValueError("Gradient DWI must be four-dimensional")
    bvals = np.loadtxt(bval_path, ndmin=1).reshape(-1)
    bvecs = np.loadtxt(bvec_path, ndmin=2)
    if bvals.size != image.shape[3] or image.shape[3] not in bvecs.shape:
        raise ValueError("DWI gradient tables do not match the number of volumes")
    rounded = np.round(bvals / 50) * 50
    shells = sorted(float(value) for value in np.unique(rounded))
    data = np.asarray(image.dataobj, dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    channels: list[dict[str, Any]] = []
    for shell in shells:
        indices = np.where(rounded == shell)[0]
        mean_image = nib.Nifti1Image(
            np.mean(data[..., indices], axis=3), image.affine, image.header
        )
        native = output_dir / f"{_stem(source_path)}_shell-{int(shell)}_mean.nii.gz"
        nib.save(mean_image, str(native))
        output = output_dir / f"{_stem(source_path)}_shell-{int(shell)}_space-common.nii.gz"
        _resample_image(mean_image, target_image, output, 1)
        channels.append(
            {
                "shell_b": shell,
                "volumes": int(indices.size),
                "path": str(output),
                "grid_id": grid_spec(output).grid_id,
            }
        )
        native.unlink(missing_ok=True)
    return {
        "present": True,
        "payload_kind": "dwi-shell-scalar-maps",
        "raw_gradient_image": str(source_path),
        "raw_resampled": False,
        "gradient_safe": True,
        "shell_channels": channels,
        "grid_id": grid_spec(channels[0]["path"]).grid_id if channels else None,
        "stackable": bool(channels),
    }


def _align_sensor(
    sample: dict[str, Any],
    output_dir: Path,
    transform_root: Path | None,
    participant: str,
    visit_id: str,
    require_standard: bool,
) -> dict[str, Any]:
    auxiliaries = sample.get("auxiliaries") if isinstance(sample.get("auxiliaries"), dict) else {}
    electrodes_value = auxiliaries.get("electrodes")
    coordinates_value = auxiliaries.get("coordsystem")
    if not coordinates_value:
        if require_standard:
            raise FileNotFoundError("coordsystem sidecar is required")
        return {**sample, "present": True, "coordinate_status": "sensor-native", "stackable": False}
    coordsystem = _read_object(Path(str(coordinates_value)))
    coordinate_name = " ".join(
        str(value) for key, value in coordsystem.items() if "CoordinateSystem" in key
    )
    standard = any(token in coordinate_name.lower() for token in ("mni", "talairach", "acpc"))
    if electrodes_value and Path(str(electrodes_value)).exists():
        electrode_path = Path(str(electrodes_value))
        rows = _read_tsv(electrode_path)
        matrix = None
        if not standard:
            transform = _find_sensor_transform(transform_root, participant, visit_id)
            if transform:
                matrix = np.loadtxt(transform)
                if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                    raise ValueError(f"Invalid sensor coordinate transform: {transform}")
            elif require_standard:
                raise FileNotFoundError("sensor-to-standard coordinate transform is required")
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / electrode_path.name
        if matrix is not None:
            _transform_coordinate_rows(rows, matrix)
            coordinate_name = "MNI152NLin2009cAsym"
        _write_tsv(rows, output)
        copied_coords = output_dir / Path(str(coordinates_value)).name
        updated_coordsystem = dict(coordsystem)
        updated_coordsystem["NeuroPrepCoordinateStatus"] = (
            "transformed-to-standard" if matrix is not None else "validated-existing"
        )
        updated_coordsystem["NeuroPrepCoordinateReference"] = coordinate_name
        _write_json(copied_coords, updated_coordsystem)
        return {
            **sample,
            "present": True,
            "coordinate_status": updated_coordsystem["NeuroPrepCoordinateStatus"],
            "coordinate_reference": coordinate_name,
            "electrodes": str(output),
            "coordsystem": str(copied_coords),
            "stackable": False,
        }
    if require_standard and not standard:
        raise FileNotFoundError("electrode coordinates are required for standard alignment")
    return {
        **sample,
        "present": True,
        "coordinate_status": "validated-existing" if standard else "sensor-native",
        "coordinate_reference": coordinate_name,
        "stackable": False,
    }


def _resample_image(source: Any, target: Any, output: Path, order: int) -> None:
    target_grid = (target.shape[:3], target.affine)
    if len(source.shape) == 3:
        result = resample_from_to(source, target_grid, order=order, mode="constant", cval=0.0)
        nib.save(result, str(output))
        return
    volumes = []
    for index in range(source.shape[3]):
        volume = nib.Nifti1Image(
            np.asarray(source.dataobj[..., index]), source.affine, source.header
        )
        resampled = resample_from_to(volume, target_grid, order=order, mode="constant", cval=0.0)
        volumes.append(np.asarray(resampled.dataobj, dtype=np.float32))
    data = np.stack(volumes, axis=3)
    nib.save(nib.Nifti1Image(data, target.affine, target.header), str(output))


def _apply_mask(image_path: Path, mask_path: Path, modality: str) -> None:
    image = nib.load(str(image_path))
    mask = np.asarray(nib.load(str(mask_path)).dataobj) > 0.5
    data = np.asarray(image.dataobj, dtype=np.float32)
    fill = -1024.0 if modality == "ct" else 0.0
    if data.ndim == 3:
        data[~mask] = fill
    else:
        data[~mask, :] = fill
    nib.save(nib.Nifti1Image(data, image.affine, image.header), str(image_path))


def _interpolation_order(modality: str) -> int:
    return 1 if modality in {"ct", "pet", "asl"} else 3


def _aligned_name(name: str, grid_id: str) -> str:
    stem = name[:-7] if name.endswith(".nii.gz") else Path(name).stem
    return f"{stem}_grid-{grid_id.removeprefix('grid-')}.nii.gz"


def _stem(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def _sidecar(path: Path) -> Path:
    return path.with_name(_stem(path) + ".json")


def _atomic_copy(source: Path, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def _find_sensor_transform(root: Path | None, participant: str, visit_id: str) -> Path | None:
    if root is None or not root.exists():
        return None
    patterns = (
        f"sub-{participant}_{visit_id}_from-sensor_to-MNI*_xfm.txt",
        f"sub-{participant}*_from-sensor_to-MNI*_xfm.txt",
    )
    candidates = sorted({path for pattern in patterns for path in root.rglob(pattern)})
    return candidates[0] if len(candidates) == 1 else None


def _transform_coordinate_rows(rows: list[dict[str, str]], matrix: np.ndarray) -> None:
    for row in rows:
        try:
            coordinate = np.array(
                [float(row[axis]) for axis in ("x", "y", "z")] + [1.0], dtype=float
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Electrode rows require finite numeric x/y/z coordinates") from exc
        transformed = matrix @ coordinate
        if not np.isfinite(transformed[:3]).all():
            raise ValueError("Coordinate transform produced non-finite values")
        for axis, value in zip(("x", "y", "z"), transformed[:3], strict=True):
            row[axis] = f"{float(value):.8g}"


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _write_tsv(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        raise ValueError("Coordinate table is empty")
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
