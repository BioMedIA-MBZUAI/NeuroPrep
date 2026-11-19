from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


@dataclass
class ImageQC:
    path: str
    shape: tuple[int, ...] | None = None
    zooms: tuple[float, ...] | None = None
    orientation: tuple[str, ...] | None = None
    dtype: str | None = None
    affine_determinant: float | None = None
    finite_fraction: float | None = None
    nonzero_fraction: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    status: str = "fail"
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QCDecision:
    status: str
    reports: list[dict[str, Any]]
    reasons: list[str] = field(default_factory=list)
    visual_review_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_image(path: Path, sample_axis_size: int = 64) -> ImageQC:
    report = ImageQC(path=str(path))
    try:
        image = nib.load(str(path))
    except Exception as exc:  # nibabel raises several format-specific errors
        report.flags.append(f"unreadable: {exc}")
        return report

    report.shape = tuple(int(value) for value in image.shape)
    report.zooms = tuple(float(value) for value in image.header.get_zooms()[: len(image.shape)])
    report.orientation = tuple(str(value) for value in nib.aff2axcodes(image.affine))
    report.dtype = str(image.get_data_dtype())
    report.affine_determinant = float(np.linalg.det(image.affine[:3, :3]))

    if len(image.shape) not in {3, 4}:
        report.flags.append(f"unexpected dimensionality: {len(image.shape)}")
    if any(size < 2 for size in image.shape[:3]):
        report.flags.append("degenerate spatial dimension")
    if any(not math.isfinite(zoom) or zoom <= 0 for zoom in report.zooms[:3]):
        report.flags.append("invalid voxel spacing")
    if not math.isfinite(report.affine_determinant) or abs(report.affine_determinant) < 1e-8:
        report.flags.append("singular or invalid affine")

    slices = tuple(
        slice(None, None, max(1, int(size) // sample_axis_size)) for size in image.shape
    )
    try:
        sample = np.asanyarray(image.dataobj[slices], dtype=np.float32)
        finite = np.isfinite(sample)
        report.finite_fraction = float(finite.mean()) if sample.size else 0.0
        if finite.any():
            finite_values = sample[finite]
            report.nonzero_fraction = float(np.count_nonzero(finite_values) / finite_values.size)
            report.min_value = float(finite_values.min())
            report.max_value = float(finite_values.max())
        else:
            report.nonzero_fraction = 0.0
        if report.finite_fraction < 1.0:
            report.flags.append("non-finite voxels")
        if not report.nonzero_fraction:
            report.flags.append("image is empty")
    except Exception as exc:
        report.flags.append(f"voxel sampling failed: {exc}")

    report.status = "pass" if not report.flags else "fail"
    return report


def normalized_mutual_information(fixed: Path, moving: Path, bins: int = 64) -> float:
    fixed_image = nib.load(str(fixed))
    moving_image = nib.load(str(moving))
    if fixed_image.shape[:3] != moving_image.shape[:3]:
        raise ValueError("Registration QC requires images on the same grid")
    fixed_data = _sample_3d(fixed_image)
    moving_data = _sample_3d(moving_image)
    valid = np.isfinite(fixed_data) & np.isfinite(moving_data)
    fixed_values = fixed_data[valid]
    moving_values = moving_data[valid]
    if fixed_values.size < 100:
        raise ValueError("Not enough finite voxels for registration QC")
    histogram, _, _ = np.histogram2d(fixed_values, moving_values, bins=bins)
    probability = histogram / histogram.sum()
    px = probability.sum(axis=1)
    py = probability.sum(axis=0)
    hxy = _entropy(probability)
    if hxy == 0:
        return 0.0
    return float((_entropy(px) + _entropy(py)) / hxy)


def _sample_3d(image: nib.spatialimages.SpatialImage, size: int = 96) -> np.ndarray:
    slices = tuple(slice(None, None, max(1, int(dim) // size)) for dim in image.shape[:3])
    return np.asanyarray(image.dataobj[slices], dtype=np.float32)


def _entropy(probability: np.ndarray) -> float:
    nonzero = probability[probability > 0]
    return float(-(nonzero * np.log(nonzero)).sum())


def write_qc(reports: list[ImageQC], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "pass" if all(report.status == "pass" for report in reports) else "fail",
        "images": [report.to_dict() for report in reports],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def aggregate_qc_reports(report_paths: list[Path], required: list[str] | None = None) -> QCDecision:
    required_names = set(required or [])
    reports: list[dict[str, Any]] = []
    reasons: list[str] = []
    seen: set[str] = set()
    has_failure = False
    needs_review = False
    for path in report_paths:
        name = path.stem
        seen.add(name)
        if not path.exists():
            reports.append({"name": name, "path": str(path), "status": "missing"})
            reasons.append(f"missing-report:{name}")
            has_failure = True
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            reports.append(
                {"name": name, "path": str(path), "status": "invalid", "error": str(exc)}
            )
            reasons.append(f"invalid-report:{name}")
            has_failure = True
            continue
        raw_status = str(payload.get("status", "review")).lower()
        status = raw_status if raw_status in {"pass", "fail", "review"} else "review"
        reports.append({"name": name, "path": str(path), "status": status, "payload": payload})
        if status == "fail":
            has_failure = True
            reasons.append(f"failed-report:{name}")
        elif status == "review":
            needs_review = True
            reasons.append(f"review-report:{name}")
    for missing in sorted(required_names - seen):
        reports.append({"name": missing, "path": None, "status": "missing"})
        reasons.append(f"missing-required-report:{missing}")
        has_failure = True
    status = "fail" if has_failure else "review" if needs_review else "pass"
    return QCDecision(status, reports, sorted(set(reasons)), needs_review)


def write_qc_decision(decision: QCDecision, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(decision.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
