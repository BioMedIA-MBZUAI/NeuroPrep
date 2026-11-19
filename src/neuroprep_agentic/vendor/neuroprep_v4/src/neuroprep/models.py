from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Modality(str, Enum):
    T1W = "t1w"
    T2W = "t2w"
    FLAIR = "flair"
    FMRI = "fmri"
    DWI = "dwi"
    PET = "pet"
    CT = "ct"
    SWI = "swi"
    MRA = "mra"
    ASL = "asl"
    EEG = "eeg"
    MEG = "meg"
    IEEG = "ieeg"
    FIELDMAP = "fieldmap"
    PD = "pd"
    UNKNOWN = "unknown"


class Space(str, Enum):
    NATIVE = "native"
    SUBJECT = "subject-anat"
    LONGITUDINAL = "subject-longitudinal"
    STANDARD = "MNI152NLin2009cAsym"


class TaskStage(str, Enum):
    INGEST = "ingest"
    VALIDATE = "validate"
    PREPROCESS = "preprocess"
    DENOISE = "denoise"
    POSTPROCESS = "postprocess"
    REGISTER = "register"
    QC = "qc"
    RELEASE = "release"
    INDEX = "index"
    GRID = "grid"
    ASSEMBLE = "assemble"
    ALIGN = "align"
    TENSOR = "tensor"
    LINK = "link"
    SPLIT = "split"
    HARMONIZE = "harmonize"
    AUDIT = "audit"


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    root: Path
    bids_root: Path
    derivatives_root: Path
    work_root: Path
    adapter: str = "bids"
    input_layout: str = "bids"
    subject_regex: str | None = None
    session_regex: str | None = None
    exclude: tuple[str, ...] = ("derivatives", ".git", "logs")
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    modality: Modality
    subject: str
    format: str = "nifti"
    session: str | None = None
    suffix: str | None = None
    sidecar: Path | None = None
    is_derivative: bool = False
    entities: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path)
        result["modality"] = self.modality.value
        result["sidecar"] = str(self.sidecar) if self.sidecar else None
        return result


@dataclass
class Inventory:
    dataset_id: str
    root: Path
    images: list[ImageRecord]
    warnings: list[str] = field(default_factory=list)

    @property
    def subjects(self) -> list[str]:
        return sorted({image.subject for image in self.images})

    def modalities(self) -> dict[Modality, int]:
        counts = {modality: 0 for modality in Modality}
        for image in self.images:
            counts[image.modality] += 1
        return {key: value for key, value in counts.items() if value}

    def subject_modality_counts(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for image in self.images:
            subject_counts = counts.setdefault(image.subject, {})
            key = image.modality.value
            subject_counts[key] = subject_counts.get(key, 0) + 1
        return {
            subject: dict(sorted(modalities.items()))
            for subject, modalities in sorted(counts.items())
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "root": str(self.root),
            "subjects": self.subjects,
            "modality_counts": {key.value: value for key, value in self.modalities().items()},
            "subject_modality_counts": self.subject_modality_counts(),
            "session_count": len(
                {
                    (image.subject, image.session or "single")
                    for image in self.images
                    if image.subject != "__unknown__"
                }
            ),
            "format_counts": {
                name: sum(image.format == name for image in self.images)
                for name in sorted({image.format for image in self.images})
            },
            "warnings": self.warnings,
            "images": [image.to_dict() for image in self.images],
        }


@dataclass
class Task:
    task_id: str
    dataset_id: str
    stage: TaskStage
    pipeline: str
    subject: str | None = None
    session: str | None = None
    dependencies: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["stage"] = self.stage.value
        return result


@dataclass
class Plan:
    version: str
    dataset_id: str
    generated_at: str
    tasks: list[Task]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dataset_id": self.dataset_id,
            "generated_at": self.generated_at,
            "warnings": self.warnings,
            "tasks": [task.to_dict() for task in self.tasks],
        }
