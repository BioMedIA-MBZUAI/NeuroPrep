from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml

from .qc import normalized_mutual_information


DEFAULT_THRESHOLDS: dict[str, Any] = {
    "global": {
        "finite_fraction_min": 1.0,
        "brain_fraction_min": 0.03,
        "brain_fraction_max": 0.85,
        "registration_dice_min": 0.70,
        "registration_nmi_min": 1.02,
    },
    "mri": {"snr_min": 3.0, "fber_min": 1.5, "efc_max": 0.95},
    "fmri": {
        "mean_fd_max": 0.30,
        "high_motion_fraction_max": 0.30,
        "dvars_outlier_fraction_max": 0.20,
    },
    "dwi": {"mean_fd_max": 1.0, "outlier_slice_fraction_max": 0.10},
    "pet": {"mean_fd_max": 3.0, "negative_fraction_max": 0.01},
    "asl": {"negative_fraction_max": 0.25, "extreme_fraction_max": 0.10},
    "qsm": {"extreme_fraction_max": 0.05},
    "ct": {"minimum_hu": -1100.0, "maximum_hu": 4000.0},
    "sensor": {
        "bad_channel_fraction_max": 0.20,
        "line_noise_ratio_max": 0.35,
        "flat_channel_fraction_max": 0.05,
    },
}


@dataclass(frozen=True)
class QualityDecision:
    status: str
    modality: str
    path: str
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failures"] = list(self.failures)
        payload["warnings"] = list(self.warnings)
        return payload


def production_qc(
    finalization_markers: list[Path],
    output: Path,
    thresholds_path: Path | None = None,
    review_policy: str = "block",
) -> dict[str, Any]:
    if review_policy not in {"block", "advisory"}:
        raise ValueError("review_policy must be 'block' or 'advisory'")
    thresholds = _load_thresholds(thresholds_path)
    decisions: list[QualityDecision] = []
    errors: list[str] = []
    for marker in finalization_markers:
        try:
            payload = _read_object(marker)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        modality = str(payload.get("modality", "unknown"))
        for record in payload.get("records", []):
            if not isinstance(record, dict):
                errors.append(f"invalid-finalization-record:{marker}")
                continue
            try:
                decisions.append(evaluate_record(record, modality, thresholds))
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                errors.append(f"{modality}:{record.get('output')}:{exc}")
    has_failed_records = any(decision.status == "fail" for decision in decisions)
    releasable_records = sum(decision.status != "fail" for decision in decisions)
    has_review = any(decision.status == "review" for decision in decisions)
    status = (
        "fail"
        if errors or (has_failed_records and not releasable_records)
        else "review"
        if has_review and review_policy == "block"
        else "partial"
        if has_failed_records
        else "pass"
    )
    summary = {
        "contract_version": "3.0",
        "status": status,
        "review_policy": review_policy,
        "thresholds": thresholds,
        "records": [decision.to_dict() for decision in decisions],
        "errors": errors,
        "counts": {
            name: sum(decision.status == name for decision in decisions)
            for name in ("pass", "review", "fail")
        },
        "releasable_records": releasable_records,
        "excluded_records": sum(
            decision.status == "fail" for decision in decisions
        ),
        "visual_review_required": status == "review",
        "visual_review_recommended": has_review,
    }
    _write_json(output, summary)
    return summary


def evaluate_record(
    record: dict[str, Any], modality: str, thresholds: dict[str, Any] | None = None
) -> QualityDecision:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    path = Path(str(record.get("output", "")))
    if not path.exists():
        raise FileNotFoundError(path)
    if modality in {"eeg", "meg", "ieeg"}:
        metrics = sensor_metrics(path)
        return _sensor_decision(path, modality, metrics, thresholds)
    mask_value = record.get("mask")
    mask_path = Path(str(mask_value)) if mask_value else None
    metrics = image_quality_metrics(path, mask_path)
    metadata = _read_optional_object(record.get("metadata"))
    failures: list[str] = []
    warnings: list[str] = []
    global_thresholds = thresholds["global"]
    _lower(metrics, "finite_fraction", global_thresholds["finite_fraction_min"], failures)
    _range(
        metrics,
        "brain_fraction",
        global_thresholds["brain_fraction_min"],
        global_thresholds["brain_fraction_max"],
        failures,
    )
    if modality in {"t1w", "t2w", "flair", "pd", "mra", "swi"}:
        policy = thresholds["mri"]
        _lower(metrics, "snr", policy["snr_min"], warnings)
        _lower(metrics, "fber", policy["fber_min"], warnings)
        _upper(metrics, "efc", policy["efc_max"], warnings)
    auxiliaries = record.get("auxiliaries") if isinstance(record.get("auxiliaries"), dict) else {}
    confounds = auxiliaries.get("confounds")
    if confounds and Path(str(confounds)).exists():
        motion = motion_metrics(Path(str(confounds)))
        metrics["motion"] = motion
        policy = thresholds.get(modality, thresholds.get("fmri", {}))
        if "mean_fd_max" in policy:
            _upper(motion, "mean_fd", policy["mean_fd_max"], failures)
        if "high_motion_fraction_max" in policy:
            _upper(
                motion,
                "high_motion_fraction",
                policy["high_motion_fraction_max"],
                failures,
            )
        if "dvars_outlier_fraction_max" in policy:
            _upper(
                motion,
                "dvars_outlier_fraction",
                policy["dvars_outlier_fraction_max"],
                warnings,
            )
        if "outlier_slice_fraction_max" in policy:
            _upper(
                motion,
                "outlier_fraction",
                policy["outlier_slice_fraction_max"],
                failures,
            )
    _quantitative_checks(path, modality, metadata, metrics, failures, warnings, thresholds)
    registration = record.get("registration_reference")
    reference_mask = record.get("registration_reference_mask")
    if registration and reference_mask and mask_path:
        metrics["registration"] = registration_metrics(
            path, Path(str(registration)), mask_path, Path(str(reference_mask))
        )
        _lower(
            metrics["registration"],
            "dice",
            global_thresholds["registration_dice_min"],
            failures,
        )
        _lower(
            metrics["registration"],
            "normalized_mutual_information",
            global_thresholds["registration_nmi_min"],
            warnings,
        )
    status = "fail" if failures else "review" if warnings else "pass"
    return QualityDecision(
        status,
        modality,
        str(path),
        metrics,
        tuple(sorted(set(failures))),
        tuple(sorted(set(warnings))),
    )


def image_quality_metrics(path: Path, mask_path: Path | None) -> dict[str, Any]:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    spatial = np.mean(data, axis=3) if data.ndim == 4 else data
    finite = np.isfinite(spatial)
    if mask_path and mask_path.exists():
        mask = np.asarray(nib.load(str(mask_path)).dataobj) > 0.5
        if mask.shape != spatial.shape:
            raise ValueError("QC mask and image grids differ")
    else:
        mask = finite & (spatial != 0)
    foreground = spatial[mask & finite]
    background = spatial[(~mask) & finite]
    if foreground.size < 10:
        raise ValueError("QC brain mask contains too few finite voxels")
    center = float(np.median(foreground))
    noise = float(np.median(np.abs(foreground - center)) * 1.4826)
    snr = abs(center) / noise if noise > np.finfo(float).eps else math.inf
    foreground_energy = float(np.mean(np.square(foreground, dtype=np.float64)))
    background_energy = (
        float(np.mean(np.square(background, dtype=np.float64))) if background.size else 0.0
    )
    fber = foreground_energy / max(background_energy, np.finfo(float).eps)
    absolute = np.abs(spatial[finite]).astype(np.float64)
    norm = float(np.sqrt(np.square(absolute).sum()))
    if norm > 0 and absolute.size > 1:
        probability = absolute / norm
        entropy = float(-np.sum(probability * np.log(probability + np.finfo(float).eps)))
        maximum_entropy = float(np.sqrt(probability.size) * np.log(np.sqrt(probability.size)))
        efc = entropy / maximum_entropy
    else:
        efc = 1.0
    return {
        "shape": list(image.shape),
        "voxel_sizes_mm": [float(value) for value in image.header.get_zooms()[:3]],
        "orientation": list(nib.aff2axcodes(image.affine)),
        "finite_fraction": float(np.mean(finite)),
        "brain_fraction": float(np.mean(mask)),
        "snr": float(snr),
        "fber": float(fber),
        "efc": efc,
        "foreground_median": center,
        "foreground_mad_sigma": noise,
        "foreground_p01": float(np.percentile(foreground, 1)),
        "foreground_p99": float(np.percentile(foreground, 99)),
    }


def registration_metrics(
    fixed: Path, moving: Path, fixed_mask: Path, moving_mask: Path
) -> dict[str, float]:
    fixed_mask_data = np.asarray(nib.load(str(fixed_mask)).dataobj) > 0.5
    moving_mask_data = np.asarray(nib.load(str(moving_mask)).dataobj) > 0.5
    if fixed_mask_data.shape != moving_mask_data.shape:
        raise ValueError("Registration masks must share a grid")
    denominator = int(fixed_mask_data.sum() + moving_mask_data.sum())
    dice = (
        2.0 * float(np.logical_and(fixed_mask_data, moving_mask_data).sum()) / denominator
        if denominator
        else 0.0
    )
    return {
        "dice": dice,
        "normalized_mutual_information": normalized_mutual_information(fixed, moving),
    }


def motion_metrics(path: Path, fd_threshold: float = 0.3) -> dict[str, float]:
    rows = _read_table(path)
    if not rows:
        raise ValueError(f"Motion/confounds table is empty: {path}")
    columns = rows[0].keys()
    fd_column = next(
        (name for name in columns if name.lower() in {"framewise_displacement", "fd", "mean_fd"}),
        None,
    )
    fd = _column(rows, fd_column) if fd_column else np.array([], dtype=float)
    dvars_columns = [name for name in columns if "dvars" in name.lower()]
    dvars = _column(rows, dvars_columns[0]) if dvars_columns else np.array([], dtype=float)
    outlier_columns = [name for name in columns if "outlier" in name.lower()]
    outliers = (
        np.column_stack([_column(rows, name) for name in outlier_columns])
        if outlier_columns
        else np.empty((len(rows), 0))
    )
    finite_fd = fd[np.isfinite(fd)]
    return {
        "mean_fd": float(np.mean(finite_fd)) if finite_fd.size else 0.0,
        "max_fd": float(np.max(finite_fd)) if finite_fd.size else 0.0,
        "high_motion_fraction": float(np.mean(finite_fd > fd_threshold)) if finite_fd.size else 0.0,
        "dvars_outlier_fraction": float(np.mean(np.abs(dvars[np.isfinite(dvars)]) > 2.5))
        if np.isfinite(dvars).any()
        else 0.0,
        "outlier_fraction": float(np.mean(outliers > 0)) if outliers.size else 0.0,
    }


def sensor_metrics(path: Path) -> dict[str, Any]:
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("Sensor QC requires NeuroPrep's sensors extra") from exc
    raw = mne.io.read_raw_fif(path, preload=True, verbose="error")
    picks = mne.pick_types(raw.info, meg=True, eeg=True, seeg=True, ecog=True, exclude=[])
    if not len(picks):
        raise ValueError("No sensor data channels found")
    data = raw.get_data(picks=picks, reject_by_annotation="omit")
    variances = np.var(data, axis=1)
    flat = variances <= np.finfo(float).eps
    spectrum = np.abs(np.fft.rfft(data, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(data.shape[1], 1 / float(raw.info["sfreq"]))
    line_frequency = float(raw.info.get("line_freq") or 50.0)
    line_band = np.abs(frequencies - line_frequency) <= 1.0
    broad_band = (frequencies >= 1.0) & (frequencies <= min(100.0, frequencies[-1]))
    line_ratio = float(
        spectrum[:, line_band].sum() / max(spectrum[:, broad_band].sum(), np.finfo(float).eps)
    )
    return {
        "channels": int(len(picks)),
        "samples": int(data.shape[1]),
        "sampling_frequency_hz": float(raw.info["sfreq"]),
        "bad_channels": list(raw.info.get("bads", [])),
        "bad_channel_fraction": len(raw.info.get("bads", [])) / len(picks),
        "flat_channel_fraction": float(np.mean(flat)),
        "line_noise_ratio": line_ratio,
        "duration_seconds": float(data.shape[1] / raw.info["sfreq"]),
    }


def _sensor_decision(
    path: Path, modality: str, metrics: dict[str, Any], thresholds: dict[str, Any]
) -> QualityDecision:
    failures: list[str] = []
    warnings: list[str] = []
    policy = thresholds["sensor"]
    _upper(metrics, "bad_channel_fraction", policy["bad_channel_fraction_max"], failures)
    _upper(metrics, "flat_channel_fraction", policy["flat_channel_fraction_max"], failures)
    _upper(metrics, "line_noise_ratio", policy["line_noise_ratio_max"], warnings)
    status = "fail" if failures else "review" if warnings else "pass"
    return QualityDecision(status, modality, str(path), metrics, tuple(failures), tuple(warnings))


def _quantitative_checks(
    path: Path,
    modality: str,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    failures: list[str],
    warnings: list[str],
    thresholds: dict[str, Any],
) -> None:
    data = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if not finite.size:
        failures.append("no-finite-values")
        return
    if modality == "ct":
        if str(metadata.get("Units", "")).upper() != "HU":
            failures.append("ct-units-not-hu")
        _range(
            {"range": float(np.min(finite))},
            "range",
            thresholds["ct"]["minimum_hu"],
            thresholds["ct"]["maximum_hu"],
            warnings,
        )
        if float(np.max(finite)) > thresholds["ct"]["maximum_hu"]:
            warnings.append("ct-maximum-hu-out-of-range")
    if modality in {"pet", "asl"}:
        metrics["negative_fraction"] = float(np.mean(finite < 0))
        _upper(
            metrics,
            "negative_fraction",
            thresholds[modality]["negative_fraction_max"],
            warnings,
        )
    if modality == "asl":
        metrics["extreme_fraction"] = float(np.mean(np.abs(finite) > 200))
        _upper(metrics, "extreme_fraction", thresholds["asl"]["extreme_fraction_max"], warnings)
    if modality == "swi" and any(
        token in str(metadata).lower() for token in ("qsm", "ppm", "susceptibility")
    ):
        center = float(np.median(finite))
        scale = float(np.median(np.abs(finite - center)) * 1.4826) or 1.0
        metrics["extreme_fraction"] = float(np.mean(np.abs(finite - center) > 8 * scale))
        _upper(metrics, "extreme_fraction", thresholds["qsm"]["extreme_fraction_max"], warnings)


def _load_thresholds(path: Path | None) -> dict[str, Any]:
    payload = json.loads(json.dumps(DEFAULT_THRESHOLDS))
    if path is None:
        return payload
    custom = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(custom, dict):
        raise ValueError("QC thresholds file must contain an object")
    for section, values in custom.items():
        if not isinstance(values, dict):
            raise ValueError(f"QC threshold section must be an object: {section}")
        payload.setdefault(section, {}).update(values)
    return payload


def _lower(metrics: dict[str, Any], key: str, threshold: float, reasons: list[str]) -> None:
    value = metrics.get(key)
    if value is None or not math.isfinite(float(value)) or float(value) < threshold:
        reasons.append(f"{key}-below-{threshold}")


def _upper(metrics: dict[str, Any], key: str, threshold: float, reasons: list[str]) -> None:
    value = metrics.get(key)
    if value is None or not math.isfinite(float(value)) or float(value) > threshold:
        reasons.append(f"{key}-above-{threshold}")


def _range(
    metrics: dict[str, Any], key: str, minimum: float, maximum: float, reasons: list[str]
) -> None:
    value = metrics.get(key)
    if value is None or not math.isfinite(float(value)) or not minimum <= float(value) <= maximum:
        reasons.append(f"{key}-outside-{minimum}-{maximum}")


def _read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def _column(rows: list[dict[str, str]], name: str | None) -> np.ndarray:
    if name is None:
        return np.array([], dtype=float)
    values = []
    for row in rows:
        try:
            values.append(float(row[name]))
        except (KeyError, TypeError, ValueError):
            values.append(float("nan"))
    return np.asarray(values, dtype=float)


def _read_optional_object(value: Any) -> dict[str, Any]:
    if not value or not Path(str(value)).exists():
        return {}
    return _read_object(Path(str(value)))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON report: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
