from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..classification import classify_series, infer_suffix
from ..models import DatasetConfig, ImageRecord, Modality


def _sidecar_for(path: Path) -> Path | None:
    if path.name.endswith(".nii.gz"):
        candidate = path.with_suffix("").with_suffix(".json")
    else:
        candidate = path.with_suffix(".json")
    return candidate if candidate.exists() else None


def _read_sidecar(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _entities(path: Path) -> dict[str, str]:
    name = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    entities: dict[str, str] = {}
    for token in name.split("_"):
        if "-" not in token:
            continue
        key, value = token.split("-", 1)
        if key and value:
            entities[key] = value
    return entities


class DatasetAdapter(ABC):
    def __init__(self, config: DatasetConfig):
        self.config = config

    @abstractmethod
    def identify(self, path: Path) -> tuple[str, str | None]:
        """Return subject and optional session identifiers."""

    def image_record(self, path: Path, image_format: str = "nifti") -> ImageRecord:
        subject, session = self.identify(path)
        sidecar = _sidecar_for(path)
        metadata = _read_sidecar(sidecar)
        entities = _entities(path)
        return ImageRecord(
            path=path,
            modality=classify_series(path, metadata),
            subject=subject,
            format=image_format,
            session=session,
            suffix=infer_suffix(path),
            sidecar=sidecar,
            is_derivative="derivatives" in path.parts,
            entities=entities,
        )


class BidsAdapter(DatasetAdapter):
    def identify(self, path: Path) -> tuple[str, str | None]:
        entities = _entities(path)
        subject = entities.get("sub")
        session = entities.get("ses")
        if subject is None:
            subject_match = re.search(r"(?:^|/)sub-([A-Za-z0-9]+)(?:/|_)", path.as_posix())
            subject = subject_match.group(1) if subject_match else "__unknown__"
        return subject, session


class OasisAdapter(DatasetAdapter):
    _subject = re.compile(r"(OAS[34]\d+)", re.I)
    _experiment = re.compile(r"(OAS[34]\d+_[A-Za-z0-9]+_d\d+)", re.I)

    def identify(self, path: Path) -> tuple[str, str | None]:
        text = path.as_posix()
        subject_match = self._subject.search(text)
        experiment_match = self._experiment.search(text)
        return (
            subject_match.group(1).upper() if subject_match else "__unknown__",
            experiment_match.group(1) if experiment_match else None,
        )


class ScanAdapter(DatasetAdapter):
    _subject = re.compile(r"(NACC\d+)", re.I)
    _image_id = re.compile(r"(?:^|[_/])(I\d+)(?:[_/.]|$)", re.I)

    def identify(self, path: Path) -> tuple[str, str | None]:
        text = path.as_posix()
        subject_match = self._subject.search(text)
        return (
            subject_match.group(1).upper() if subject_match else "__unknown__",
            None,
        )

    def image_record(self, path: Path, image_format: str = "nifti") -> ImageRecord:
        record = super().image_record(path, image_format)
        image_match = self._image_id.search(path.as_posix())
        entities = dict(record.entities)
        if image_match:
            entities["loniuid"] = image_match.group(1).upper()
        directories = {part.upper() for part in path.parts[:-1]}
        modality = record.modality
        if "PET" in directories:
            modality = Modality.PET
        elif "FMRI" in directories:
            modality = Modality.FMRI
        return replace(record, modality=modality, entities=entities)


class GenericAdapter(DatasetAdapter):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.subject_pattern = re.compile(config.subject_regex) if config.subject_regex else None
        self.session_pattern = re.compile(config.session_regex) if config.session_regex else None

    def identify(self, path: Path) -> tuple[str, str | None]:
        text = path.as_posix()
        subject_match = self.subject_pattern.search(text) if self.subject_pattern else None
        session_match = self.session_pattern.search(text) if self.session_pattern else None
        subject = subject_match.group(1) if subject_match else "__unknown__"
        session = session_match.group(1) if session_match else None
        return subject, session


def get_adapter(config: DatasetConfig) -> DatasetAdapter:
    adapters: dict[str, type[DatasetAdapter]] = {
        "bids": BidsAdapter,
        "oasis": OasisAdapter,
        "scan": ScanAdapter,
        "generic": GenericAdapter,
    }
    try:
        return adapters[config.adapter](config)
    except KeyError as exc:
        available = ", ".join(sorted(adapters))
        raise ValueError(f"Unknown adapter '{config.adapter}'. Available: {available}") from exc
