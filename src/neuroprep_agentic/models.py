from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SupportLevel(StrEnum):
    BIDS_CORE = "bids_core"
    BIDS_EXTENSION = "bids_extension"
    PROVISIONAL = "provisional"
    PASSTHROUGH = "passthrough"


class ClassificationStatus(StrEnum):
    RECOGNIZED = "recognized"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DatasetSpec(BaseModel):
    """Dataset configuration; no cohort name is built into the core."""

    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    adapter_id: str = "generic"
    bids_version: str = "1.11.1"
    source_roots: list[str] = Field(default_factory=list)
    required_modalities: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceRecord(BaseModel):
    """One immutable source acquisition or recording."""

    model_config = ConfigDict(extra="allow")

    dataset_id: str
    source_path: str
    source_subject_id: str
    source_session_id: str | None = None
    acquisition_time: datetime | None = None
    days_from_baseline: int | None = None
    file_format: str | None = None
    checksum: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_path", "source_subject_id")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be empty")
        return value


class ClassificationCandidate(BaseModel):
    plugin_id: str
    modality: str
    bids_datatype: str
    bids_suffix: str
    support_level: SupportLevel
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class BIDSMapping(BaseModel):
    datatype: str
    suffix: str
    extension: str
    entities: dict[str, str] = Field(default_factory=dict)
    sidecar: dict[str, Any] = Field(default_factory=dict)
    support_level: SupportLevel
    schema_id: str = "bids-core"


class AcquisitionRecord(BaseModel):
    acquisition_id: str
    source: SourceRecord
    subject_id: str
    session_id: str
    status: ClassificationStatus
    candidates: list[ClassificationCandidate] = Field(default_factory=list)
    selected_plugin_id: str | None = None
    modality: str | None = None
    bids: BIDSMapping | None = None
    issues: list[str] = Field(default_factory=list)


class SessionRecord(BaseModel):
    session_id: str
    source_session_ids: set[str] = Field(default_factory=set)
    days_from_baseline: int | None = None
    acquisition_time: datetime | None = None
    acquisitions: list[AcquisitionRecord] = Field(default_factory=list)


class SubjectRecord(BaseModel):
    subject_id: str
    dataset_id: str
    source_subject_ids: set[str]
    sessions: list[SessionRecord] = Field(default_factory=list)


class CanonicalAcquisitionGraph(BaseModel):
    schema_version: str = "1.0"
    dataset: DatasetSpec
    subjects: list[SubjectRecord] = Field(default_factory=list)
    quarantined: list[AcquisitionRecord] = Field(default_factory=list)
    unsupported: list[AcquisitionRecord] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @property
    def acquisitions(self) -> list[AcquisitionRecord]:
        return [
            acquisition
            for subject in self.subjects
            for session in subject.sessions
            for acquisition in session.acquisitions
        ]


class ValidationIssue(BaseModel):
    code: str
    severity: Severity
    message: str
    acquisition_id: str | None = None
    path: str | None = None


class ResourceRequest(BaseModel):
    resource_class: Literal["cpu", "gpu", "high_memory"] = "cpu"
    cpus: int = Field(default=1, ge=1)
    memory_gb: int = Field(default=4, ge=1)
    gpus: int = Field(default=0, ge=0)
    walltime_minutes: int = Field(default=60, ge=1)


class TaskSpec(BaseModel):
    task_id: str
    plugin_id: str
    stage: str
    subject_id: str
    session_id: str | None = None
    acquisition_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    resource: ResourceRequest = Field(default_factory=ResourceRequest)
    required: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanResult(BaseModel):
    tasks: list[TaskSpec] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)
    modality_counts: dict[str, int] = Field(default_factory=dict)


class TaskResult(BaseModel):
    task_id: str
    status: Literal["pending", "running", "complete", "failed", "blocked"]
    return_code: int | None = None
    output_paths: list[str] = Field(default_factory=list)
    qc_passed: bool | None = None


class ArtifactRecord(BaseModel):
    artifact_id: str
    kind: str
    path: str
    subject_id: str | None = None
    session_id: str | None = None
    acquisition_id: str | None = None
    size: int = Field(default=0, ge=0)
    valid: bool = False
    provenance_digest: str | None = None


class CompletionContract(BaseModel):
    required_modalities: set[str] = Field(default_factory=set)
    required_artifact_kinds: set[str] = Field(default_factory=set)
    require_all_planned_tasks: bool = True
    require_qc: bool = True
    allow_quarantine: bool = False
    allow_unsupported: bool = True


class CompletionResult(BaseModel):
    complete: bool
    reasons: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


class ActionProposal(BaseModel):
    action_type: str
    target_type: str
    target_id: str
    rationale: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    mutating: bool = True
    requires_approval: bool = True


class EvidenceEnvelope(BaseModel):
    kind: str
    source_adapter: str
    collected_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
