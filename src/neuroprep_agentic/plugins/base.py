from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from ..models import (
    BIDSMapping,
    ClassificationCandidate,
    ResourceRequest,
    SourceRecord,
    SupportLevel,
    TaskSpec,
)


@dataclass(frozen=True)
class DetectionContext:
    record: SourceRecord

    @property
    def searchable_text(self) -> str:
        values: list[str] = [self.record.source_path, self.record.file_format or ""]
        for key, value in self.record.metadata.items():
            if isinstance(value, (str, int, float)):
                values.extend((str(key), str(value)))
        return " ".join(values).lower()

    @property
    def extension(self) -> str:
        name = PurePosixPath(self.record.source_path).name.lower()
        if name.endswith(".nii.gz"):
            return ".nii.gz"
        if name.endswith(".tsv.gz"):
            return ".tsv.gz"
        if name.endswith(".ome.tif"):
            return ".ome.tif"
        return PurePosixPath(name).suffix

    @property
    def source_modality(self) -> str:
        for key in ("Modality", "modality", "DataType", "datatype"):
            value = self.record.metadata.get(key)
            if value:
                return str(value).upper()
        return ""


class DatasetAdapter(ABC):
    adapter_id: str
    aliases: tuple[str, ...] = ()

    @abstractmethod
    def normalize(self, record: SourceRecord) -> SourceRecord:
        """Normalize source metadata without altering source files."""

    def recognizes(self, dataset_id: str) -> bool:
        lowered = dataset_id.lower()
        return lowered == self.adapter_id.lower() or lowered in {
            alias.lower() for alias in self.aliases
        }


@dataclass(frozen=True)
class DeclarativeDatasetAdapter(DatasetAdapter):
    adapter_id: str
    aliases: tuple[str, ...] = ()
    subject_fields: tuple[str, ...] = ("subject_id", "Subject", "PatientID")
    session_fields: tuple[str, ...] = ("session_id", "Visit", "StudyDate")
    notes: str = ""

    def normalize(self, record: SourceRecord) -> SourceRecord:
        metadata = dict(record.metadata)
        metadata.setdefault("SourceDataset", self.adapter_id)
        return record.model_copy(update={"metadata": metadata})


class ModalityPlugin(ABC):
    plugin_id: str
    modality: str
    bids_datatype: str
    bids_suffix: str
    support_level: SupportLevel

    @abstractmethod
    def detect(self, context: DetectionContext) -> ClassificationCandidate | None:
        """Return evidence-backed detection or None."""

    @abstractmethod
    def build_bids_mapping(self, record: SourceRecord, entities: dict[str, str]) -> BIDSMapping:
        """Build a non-destructive BIDS or extension mapping."""

    @abstractmethod
    def plan_tasks(
        self,
        acquisition_id: str,
        subject_id: str,
        session_id: str,
    ) -> list[TaskSpec]:
        """Return deterministic tasks for one recognized acquisition."""


@dataclass(frozen=True)
class TaskTemplate:
    stage: str
    resource_class: str = "cpu"
    cpus: int = 2
    memory_gb: int = 8
    gpus: int = 0
    walltime_minutes: int = 60
    required: bool = True
    depends_on_previous: bool = True


@dataclass(frozen=True)
class DeclarativeModalityPlugin(ModalityPlugin):
    plugin_id: str
    modality: str
    bids_datatype: str
    bids_suffix: str
    support_level: SupportLevel
    source_modalities: tuple[str, ...] = ()
    positive_terms: tuple[str, ...] = ()
    negative_terms: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()
    schema_id: str = "bids-core"
    tasks: tuple[TaskTemplate, ...] = field(default_factory=tuple)
    default_sidecar: dict[str, Any] = field(default_factory=dict)

    def detect(self, context: DetectionContext) -> ClassificationCandidate | None:
        text = context.searchable_text
        reasons: list[str] = []
        score = 0.0

        if self.source_modalities and context.source_modality in self.source_modalities:
            score += 0.30
            reasons.append(f"source modality {context.source_modality}")

        if self.extensions and context.extension in self.extensions:
            score += 0.15
            reasons.append(f"supported extension {context.extension}")

        source_name = PurePosixPath(context.record.source_path).name.lower()
        if re.search(
            rf"(?<![a-z0-9]){re.escape(self.bids_suffix.lower())}(?![a-z0-9])",
            source_name,
        ):
            score += 0.20
            reasons.append(f"explicit BIDS suffix {self.bids_suffix}")

        matched_terms = [
            term
            for term in self.positive_terms
            if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text)
        ]
        if matched_terms:
            score += min(0.60, 0.35 + 0.10 * (len(matched_terms) - 1))
            reasons.append("matched " + ", ".join(matched_terms[:4]))

        matched_negative = [
            term
            for term in self.negative_terms
            if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text)
        ]
        if matched_negative:
            score -= 0.50
            reasons.append("excluded by " + ", ".join(matched_negative[:4]))

        score = max(0.0, min(1.0, score))
        if score == 0.0:
            return None
        return ClassificationCandidate(
            plugin_id=self.plugin_id,
            modality=self.modality,
            bids_datatype=self.bids_datatype,
            bids_suffix=self.bids_suffix,
            support_level=self.support_level,
            score=score,
            reasons=reasons,
        )

    def build_bids_mapping(self, record: SourceRecord, entities: dict[str, str]) -> BIDSMapping:
        extension = ".nii.gz"
        suffix = self.bids_suffix
        sidecar = {**self.default_sidecar}
        if self.plugin_id == "mri.fmap":
            source_name = PurePosixPath(record.source_path).name.lower()
            image_type = {str(item).upper() for item in record.metadata.get("ImageType", [])}
            if "P" in image_type or "PHASE" in image_type:
                suffix = "phasediff"
            elif "echo-1" in source_name or record.metadata.get("EchoNumber") == 1:
                suffix = "magnitude1"
            elif "echo-2" in source_name or record.metadata.get("EchoNumber") == 2:
                suffix = "magnitude2"
        if self.bids_datatype in {
            "eeg",
            "ieeg",
            "meg",
            "nirs",
            "micr",
            "x-ultrasound",
            "genetic",
        }:
            extension = source_extension(record.source_path)
        elif self.bids_datatype == "beh":
            extension = ".tsv"
        elif suffix == "physio":
            observed = source_extension(record.source_path)
            extension = ".tsv" if observed in {".csv", ".json"} else observed
        return BIDSMapping(
            datatype=self.bids_datatype,
            suffix=suffix,
            extension=extension,
            entities=entities,
            sidecar=sidecar,
            support_level=self.support_level,
            schema_id=self.schema_id,
        )

    def plan_tasks(
        self,
        acquisition_id: str,
        subject_id: str,
        session_id: str,
    ) -> list[TaskSpec]:
        planned: list[TaskSpec] = []
        prior: str | None = None
        for template in self.tasks:
            task_id = f"{subject_id}:{session_id}:{acquisition_id}:{template.stage}"
            dependencies = [prior] if template.depends_on_previous and prior else []
            planned.append(
                TaskSpec(
                    task_id=task_id,
                    plugin_id=self.plugin_id,
                    stage=template.stage,
                    subject_id=subject_id,
                    session_id=session_id,
                    acquisition_ids=[acquisition_id],
                    depends_on=dependencies,
                    resource=ResourceRequest(
                        resource_class=template.resource_class,  # type: ignore[arg-type]
                        cpus=template.cpus,
                        memory_gb=template.memory_gb,
                        gpus=template.gpus,
                        walltime_minutes=template.walltime_minutes,
                    ),
                    required=template.required,
                )
            )
            prior = task_id
        return planned


def source_extension(path: str) -> str:
    lowered = path.lower()
    for compound in (
        ".nii.gz",
        ".tsv.gz",
        ".ome.tif",
        ".vhdr",
        ".edf",
        ".bdf",
        ".set",
        ".fif",
        ".snirf",
    ):
        if lowered.endswith(compound):
            return compound
    return PurePosixPath(lowered).suffix or ".dat"
