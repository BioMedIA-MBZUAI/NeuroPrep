from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrimaryAnatomicalChoice:
    acquisition_id: str
    score: int
    reasons: tuple[str, ...]


def _text(metadata: dict[str, Any]) -> str:
    return " ".join(
        str(metadata.get(key, ""))
        for key in (
            "SeriesDescription",
            "ProtocolName",
            "SequenceName",
            "PulseSequenceDetails",
            "ScanningSequence",
            "SequenceVariant",
        )
    ).lower()


def score_primary_t1(metadata: dict[str, Any]) -> tuple[int, tuple[str, ...]]:
    """Score T1w candidates for downstream reference/segmentation use.

    The score uses acquisition facts only. It favors high-resolution 3D
    inversion-recovery scans and penalizes post-contrast, fat-suppressed, and
    thick-slice localizer-style series. It never changes the BIDS classification
    or discards secondary structural acquisitions.
    """

    score = 0
    reasons: list[str] = []
    text = _text(metadata)
    acquisition_type = str(metadata.get("MRAcquisitionType", "")).upper()
    if acquisition_type == "3D":
        score += 40
        reasons.append("3D acquisition")
    elif acquisition_type == "2D":
        score -= 20
        reasons.append("2D acquisition penalty")
    if metadata.get("InversionTime") is not None:
        score += 25
        reasons.append("inversion recovery")
    if any(marker in text for marker in ("mprage", "mp-rage", "tfl3d", "gr_ir")):
        score += 25
        reasons.append("MPRAGE-like sequence")
    thickness = metadata.get("SliceThickness")
    if isinstance(thickness, (int, float)):
        if thickness <= 1.5:
            score += 20
            reasons.append("high through-plane resolution")
        elif thickness <= 2.0:
            score += 10
            reasons.append("moderate through-plane resolution")
        elif thickness >= 4.0:
            score -= 25
            reasons.append("thick-slice penalty")
    if any(marker in text for marker in ("post", "contrast", "gd", "gad")):
        score -= 35
        reasons.append("post-contrast penalty")
    if any(marker in text for marker in ("fat sat", "fatsat", "t1fs", " t1 fs")):
        score -= 20
        reasons.append("fat-suppression penalty")
    return score, tuple(reasons)


def select_primary_t1(
    candidates: list[tuple[str, dict[str, Any]]],
) -> PrimaryAnatomicalChoice:
    if not candidates:
        raise ValueError("primary T1 selection requires at least one candidate")
    ranked: list[PrimaryAnatomicalChoice] = []
    for acquisition_id, metadata in candidates:
        score, reasons = score_primary_t1(metadata)
        ranked.append(PrimaryAnatomicalChoice(acquisition_id, score, reasons))
    return sorted(ranked, key=lambda item: (-item.score, item.acquisition_id))[0]
