from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NormalizationResult:
    data: np.ndarray
    channel_names: tuple[str, ...]
    method: str
    parameters: dict[str, Any]
    warnings: tuple[str, ...] = ()

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("data")
        payload["channel_names"] = list(self.channel_names)
        payload["warnings"] = list(self.warnings)
        return payload


def normalize_image(
    modality: str,
    data: np.ndarray,
    mask: np.ndarray | None,
    metadata: dict[str, Any] | None = None,
    reference_mask: np.ndarray | None = None,
) -> NormalizationResult:
    key = modality.lower()
    values = np.asarray(data, dtype=np.float32)
    metadata = metadata or {}
    spatial_mask = _mask(mask, values.shape[:3])
    if key in {"t1w", "t2w", "flair", "pd", "mra", "swi"}:
        if key == "swi" and _is_qsm(metadata):
            return _normalize_qsm(values, spatial_mask, reference_mask)
        normalized, parameters = _robust_z(values, spatial_mask)
        return NormalizationResult(
            normalized,
            (key,),
            "robust-brain-zscore",
            parameters,
        )
    if key == "ct":
        return _normalize_ct(values, spatial_mask, metadata)
    if key == "pet":
        return _normalize_pet(values, spatial_mask, metadata, reference_mask)
    if key == "asl":
        return _normalize_asl(values, spatial_mask, metadata)
    if key == "fmri":
        return _normalize_fmri(values, spatial_mask)
    if key == "dwi":
        return _normalize_dwi(values, spatial_mask, int(metadata.get("b0_index", 0)))
    raise ValueError(f"No image normalization policy for modality: {modality}")


def normalize_sensor(
    data: np.ndarray,
    modality: str,
    channel_types: list[str] | None = None,
) -> NormalizationResult:
    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Sensor arrays must have shape channels x time")
    channel_types = channel_types or [modality.lower()] * values.shape[0]
    if len(channel_types) != values.shape[0]:
        raise ValueError("channel_types length does not match sensor channel count")
    medians = np.nanmedian(values, axis=1, keepdims=True)
    scales = np.nanmedian(np.abs(values - medians), axis=1, keepdims=True) * 1.4826
    fallback = np.nanstd(values, axis=1, keepdims=True)
    scales = np.where(scales > np.finfo(np.float32).eps, scales, fallback)
    scales = np.where(scales > np.finfo(np.float32).eps, scales, 1.0)
    normalized = (values - medians) / scales
    normalized[~np.isfinite(normalized)] = 0
    return NormalizationResult(
        normalized.astype(np.float32),
        tuple(f"{kind}-{index:03d}" for index, kind in enumerate(channel_types)),
        "per-channel-robust-zscore",
        {
            "centers": medians[:, 0].astype(float).tolist(),
            "scales": scales[:, 0].astype(float).tolist(),
            "physical_units_preserved_in_source": True,
        },
    )


def _normalize_ct(
    data: np.ndarray, mask: np.ndarray, metadata: dict[str, Any]
) -> NormalizationResult:
    units = str(metadata.get("Units", "")).upper()
    if units != "HU":
        raise ValueError("CT tensor preparation requires verified Hounsfield units")
    if data.ndim != 3:
        raise ValueError("CT normalization expects a three-dimensional image")
    windows = (
        ("ct-brain", 40.0, 80.0),
        ("ct-subdural", 75.0, 215.0),
        ("ct-bone", 600.0, 2800.0),
    )
    channels = []
    for _, center, width in windows:
        low, high = center - width / 2, center + width / 2
        channel = np.clip(data, low, high)
        channel = (channel - low) / width
        channel[~mask] = 0
        channels.append(channel)
    return NormalizationResult(
        np.stack(channels, axis=0).astype(np.float32),
        tuple(item[0] for item in windows),
        "ct-windowing",
        {
            "windows_hu": [
                {"name": name, "center": center, "width": width}
                for name, center, width in windows
            ],
            "original_hu_preserved_in_source": True,
        },
    )


def _normalize_pet(
    data: np.ndarray,
    mask: np.ndarray,
    metadata: dict[str, Any],
    reference_mask: np.ndarray | None,
) -> NormalizationResult:
    units = str(metadata.get("Units", "")).upper()
    tracer = metadata.get("TracerName") or metadata.get("Radiopharmaceutical")
    if not units or not tracer:
        raise ValueError("PET normalization requires explicit units and tracer")
    quantitative = data.astype(np.float32, copy=True)
    method_parts = [f"source-{units.lower()}"]
    warnings: list[str] = []
    if units in {"BQML", "BQ/ML"}:
        weight = _numeric(metadata, ("BodyWeight", "PatientWeight"))
        dose = _numeric(metadata, ("InjectedRadioactivity", "RadionuclideTotalDose"))
        if weight and dose:
            quantitative = quantitative * float(weight) * 1000.0 / float(dose)
            method_parts.append("suv")
        else:
            warnings.append("SUV not computed: body weight or injected dose missing")
    if reference_mask is not None:
        reference = _mask(reference_mask, data.shape[:3])
        reference_values = quantitative[reference] if quantitative.ndim == 3 else quantitative[reference, :]
        reference_mean = float(np.nanmean(reference_values))
        if not np.isfinite(reference_mean) or abs(reference_mean) < np.finfo(float).eps:
            raise ValueError("PET reference region has no finite non-zero uptake")
        quantitative /= reference_mean
        method_parts.append("suvr")
        parameters = {
            "reference_mean": reference_mean,
            "tracer": str(tracer),
            "source_units": units,
        }
    else:
        quantitative, robust = _robust_z(quantitative, mask)
        method_parts.append("robust-brain-zscore")
        parameters = {**robust, "tracer": str(tracer), "source_units": units}
        warnings.append("No tracer-specific reference mask supplied; SUVR was not claimed")
    if quantitative.ndim == 3:
        quantitative[~mask] = 0
    else:
        quantitative[~mask, :] = 0
    return NormalizationResult(
        quantitative.astype(np.float32),
        ("pet",),
        "+".join(method_parts),
        parameters,
        tuple(warnings),
    )


def _normalize_asl(
    data: np.ndarray, mask: np.ndarray, metadata: dict[str, Any]
) -> NormalizationResult:
    units = str(metadata.get("Units") or metadata.get("QuantitativeUnits") or "")
    normalized = data.astype(np.float32, copy=True)
    warnings: list[str] = []
    if "100" in units and "min" in units.lower():
        normalized = np.clip(normalized, 0, 200) / 100.0
        method = "CBF-physiologic-clip-and-scale"
        parameters = {"clip": [0.0, 200.0], "scale": 100.0, "source_units": units}
    else:
        normalized, parameters = _robust_z(normalized, mask)
        parameters["source_units"] = units
        method = "robust-brain-zscore"
        warnings.append("CBF units unrecognized; physiologic scaling not claimed")
    if normalized.ndim == 3:
        normalized[~mask] = 0
    else:
        normalized[~mask, :] = 0
    return NormalizationResult(
        normalized.astype(np.float32), ("asl",), method, parameters, tuple(warnings)
    )


def _normalize_qsm(
    data: np.ndarray, mask: np.ndarray, reference_mask: np.ndarray | None
) -> NormalizationResult:
    reference = _mask(reference_mask, data.shape[:3]) if reference_mask is not None else mask
    values = data[reference]
    center = float(np.nanmedian(values))
    if not np.isfinite(center):
        raise ValueError("QSM reference region contains no finite values")
    normalized = data - center
    normalized[~mask] = 0
    warning = () if reference_mask is not None else (
        "QSM referenced to whole-brain median because no tissue-specific reference was supplied",
    )
    return NormalizationResult(
        normalized.astype(np.float32),
        ("qsm",),
        "reference-subtraction",
        {"reference_median": center, "reference": "provided-mask" if reference_mask is not None else "whole-brain"},
        warning,
    )


def _normalize_fmri(data: np.ndarray, mask: np.ndarray) -> NormalizationResult:
    if data.ndim != 4:
        raise ValueError("fMRI temporal normalization requires a four-dimensional image")
    try:
        from scipy.signal import detrend
    except ImportError as exc:
        raise RuntimeError("fMRI normalization requires scipy") from exc
    normalized = np.zeros_like(data, dtype=np.float32)
    timeseries = np.asarray(data[mask, :], dtype=np.float32)
    timeseries = detrend(timeseries, axis=1, type="linear")
    centers = np.mean(timeseries, axis=1, keepdims=True)
    scales = np.std(timeseries, axis=1, keepdims=True)
    scales[scales < np.finfo(np.float32).eps] = 1.0
    normalized[mask, :] = (timeseries - centers) / scales
    return NormalizationResult(
        normalized,
        ("fmri",),
        "voxelwise-linear-detrend-and-temporal-zscore",
        {"timepoints": int(data.shape[3]), "background": 0.0},
    )


def _normalize_dwi(
    data: np.ndarray, mask: np.ndarray, b0_index: int = 0
) -> NormalizationResult:
    if data.ndim == 3:
        data = data[..., np.newaxis]
    if data.ndim != 4:
        raise ValueError("DWI tensor input must be a shell-map volume or channel stack")
    if b0_index < 0 or b0_index >= data.shape[3]:
        raise ValueError("DWI b0 index is outside the shell-map channel range")
    baseline = data[..., b0_index][mask]
    scale = float(np.nanmedian(baseline))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError("DWI baseline shell has no valid positive median")
    normalized = data / scale
    normalized[~mask, :] = 0
    return NormalizationResult(
        normalized.astype(np.float32),
        tuple(f"dwi-shell-{index}" for index in range(data.shape[3])),
        "b0-median-scaling",
        {"b0_brain_median": scale, "b0_index": b0_index},
    )


def _robust_z(data: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    selected = data[mask] if data.ndim == 3 else data[mask, :]
    finite = selected[np.isfinite(selected)]
    if finite.size < 10:
        raise ValueError("Not enough finite brain values for robust normalization")
    center = float(np.median(finite))
    scale = float(np.median(np.abs(finite - center)) * 1.4826)
    if scale <= np.finfo(float).eps:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError("Image has no usable intensity variation")
    normalized = (data - center) / scale
    normalized = np.clip(normalized, -10, 10).astype(np.float32)
    if normalized.ndim == 3:
        normalized[~mask] = 0
    else:
        normalized[~mask, :] = 0
    return normalized, {"center": center, "scale": scale, "clip": [-10.0, 10.0]}


def _mask(mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        raise ValueError("A brain mask is required for model normalization")
    result = np.asarray(mask) > 0.5
    if result.shape != tuple(shape[:3]):
        raise ValueError(f"Mask shape {result.shape} does not match image shape {shape[:3]}")
    if int(result.sum()) < 10:
        raise ValueError("Brain mask is empty or implausibly small")
    return result


def _is_qsm(metadata: dict[str, Any]) -> bool:
    text = " ".join(
        str(metadata.get(field, ""))
        for field in ("Units", "QuantitativeUnits", "Description", "Sources")
    ).lower()
    return any(token in text for token in ("ppm", "qsm", "susceptibility"))


def _numeric(metadata: dict[str, Any], fields: tuple[str, ...]) -> float | None:
    for field in fields:
        try:
            value = float(metadata[field])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            return value
    return None
