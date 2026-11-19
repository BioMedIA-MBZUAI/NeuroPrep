from __future__ import annotations

from collections.abc import Iterable

from .models import ImageRecord, Modality


REFERENCE_PRIORITY = (
    Modality.T1W,
    Modality.T2W,
    Modality.FLAIR,
    Modality.CT,
)


def select_reference(images: Iterable[ImageRecord]) -> ImageRecord | None:
    candidates = list(images)
    for modality in REFERENCE_PRIORITY:
        matching = [image for image in candidates if image.modality is modality]
        if matching:
            return sorted(matching, key=_reference_score)[0]
    return None


def _reference_score(image: ImageRecord) -> tuple[int, int, str]:
    """Prefer non-derived, sidecar-backed, deterministically named references."""

    return (
        1 if image.is_derivative else 0,
        0 if image.sidecar else 1,
        image.path.as_posix(),
    )

