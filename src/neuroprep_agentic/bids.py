from __future__ import annotations

from pathlib import PurePosixPath

from .models import AcquisitionRecord, ClassificationStatus

ENTITY_ORDER = (
    "subject", "session", "task", "acquisition", "ceagent", "tracer", "direction", "run"
)


def bids_relative_path(acquisition: AcquisitionRecord) -> PurePosixPath:
    """Render a BIDS-style relative path without touching source or destination data."""

    if acquisition.status is not ClassificationStatus.RECOGNIZED or acquisition.bids is None:
        raise ValueError("Only recognized acquisitions have a BIDS path")

    entities = dict(acquisition.bids.entities)
    entities.setdefault("subject", acquisition.subject_id.removeprefix("sub-"))
    entities.setdefault("session", acquisition.session_id.removeprefix("ses-"))

    labels = []
    for key in ENTITY_ORDER:
        value = entities.get(key)
        if not value:
            continue
        short = {
            "subject": "sub",
            "session": "ses",
            "acquisition": "acq",
            "ceagent": "ce",
            "tracer": "trc",
            "direction": "dir",
        }.get(key, key)
        labels.append(f"{short}-{value}")
    labels.append(acquisition.bids.suffix)
    filename = "_".join(labels) + acquisition.bids.extension
    return (
        PurePosixPath(f"sub-{entities['subject']}")
        / f"ses-{entities['session']}"
        / acquisition.bids.datatype
        / filename
    )
