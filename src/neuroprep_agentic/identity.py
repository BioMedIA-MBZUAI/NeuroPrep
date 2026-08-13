from __future__ import annotations

import hashlib
import re
from datetime import datetime

from .models import SourceRecord


def _clean_label(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", value)
    return cleaned or fallback


class IdentityResolver:
    """Deterministic pseudonymization; cross-dataset links require an explicit crosswalk."""

    def __init__(
        self,
        namespace_key: str,
        crosswalk: dict[str, str] | None = None,
    ) -> None:
        if len(namespace_key) < 12:
            raise ValueError("namespace_key must contain at least 12 characters")
        self._key = namespace_key.encode("utf-8")
        self._crosswalk = crosswalk or {}

    def subject_id(self, dataset_id: str, source_subject_id: str) -> str:
        lookup = f"{dataset_id}:{source_subject_id}"
        if lookup in self._crosswalk:
            label = _clean_label(self._crosswalk[lookup], "linked")
            return label if label.startswith("sub") else f"sub-{label}"
        digest = hashlib.blake2b(
            lookup.encode("utf-8"),
            key=self._key,
            digest_size=8,
            person=b"neuroprep-id-v1",
        ).hexdigest()
        return f"sub-{digest}"

    def session_id(
        self,
        record: SourceRecord,
        baseline: datetime | None = None,
    ) -> str:
        days = record.days_from_baseline
        if days is None and record.acquisition_time is not None and baseline is not None:
            days = (record.acquisition_time.date() - baseline.date()).days
        if days is not None:
            prefix = "m" if days < 0 else "d"
            return f"ses-{prefix}{abs(days):05d}"
        if record.source_session_id:
            return f"ses-{_clean_label(record.source_session_id, 'unknown')}"
        return "ses-unknown"

    def acquisition_id(self, record: SourceRecord) -> str:
        material = "|".join(
            (
                record.dataset_id,
                record.source_subject_id,
                record.source_session_id or "",
                record.source_path,
                record.checksum or "",
            )
        )
        digest = hashlib.blake2b(
            material.encode("utf-8"),
            key=self._key,
            digest_size=10,
            person=b"neuroprep-acq1",
        ).hexdigest()
        return f"acq-{digest}"
