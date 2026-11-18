from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from . import paths
from .backend_evidence import BackendEvidenceBinding
from .settings import DEFAULT_AGENTIC_PROJECT, validated_cluster_path


class ExecutorBackend(StrEnum):
    BIDS_APP = "bids_app"
    CONTAINER = "container"
    NEUROPREP_V4 = "neuroprep_v4"
    NATIVE = "native"


class ExecutionScope(StrEnum):
    ACQUISITION = "acquisition"
    SESSION = "session"
    SUBJECT = "subject"
    DATASET = "dataset"


class ArtifactRequirement(BaseModel):
    kind: str
    patterns: tuple[str, ...]
    minimum_count: int = Field(default=1, ge=1)
    required: bool = True
    validator: str = "exists_nonempty"


class ExecutorDefinition(BaseModel):
    executor_id: str
    version: str
    backend: ExecutorBackend
    scope: ExecutionScope
    plugin_ids: tuple[str, ...]
    stages: tuple[str, ...]
    command: tuple[str, ...]
    image: str | None = None
    image_source: str | None = None
    executable: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    artifacts: tuple[ArtifactRequirement, ...] = ()
    required_metadata: tuple[str, ...] = ()
    requires_executor_ids: tuple[str, ...] = ()
    source: str
    notes: str = ""

    @model_validator(mode="after")
    def executable_contract(self) -> ExecutorDefinition:
        if (
            self.backend
            in {
                ExecutorBackend.BIDS_APP,
                ExecutorBackend.CONTAINER,
            }
            and not self.image
        ):
            raise ValueError("BIDS app executors require a local image path")
        if self.backend is ExecutorBackend.NEUROPREP_V4 and not self.executable:
            raise ValueError("NeuroPrep-v4 executors require an executable")
        if not self.command:
            raise ValueError("executor command cannot be empty")
        return self


class RunLayout(BaseModel):
    dataset_id: str
    source_root: str
    bids_root: str
    derivatives_root: str
    work_root: str
    state_path: str
    agentic_project: str = DEFAULT_AGENTIC_PROJECT
    neuroprep_project: str = Field(default_factory=lambda: str(paths.neuroprep_v4_project()))
    container_root: str = Field(default_factory=lambda: str(paths.container_root()))
    freesurfer_license: str | None = None
    execute_optional: bool = False

    @field_validator(
        "source_root",
        "bids_root",
        "derivatives_root",
        "work_root",
        "state_path",
        "agentic_project",
        "neuroprep_project",
        "container_root",
        "freesurfer_license",
    )
    @classmethod
    def cluster_paths_only(cls, value: str | None) -> str | None:
        if value is not None:
            validated_cluster_path(value)
        return value

    @model_validator(mode="after")
    def separated_roots(self) -> RunLayout:
        roots = {self.bids_root, self.derivatives_root, self.work_root}
        if len(roots) != 3:
            raise ValueError("BIDS, derivatives, and work roots must be distinct")
        named = {
            "source": Path(self.source_root).resolve(),
            "bids": Path(self.bids_root).resolve(),
            "derivatives": Path(self.derivatives_root).resolve(),
            "work": Path(self.work_root).resolve(),
        }
        for left, left_path in named.items():
            for right, right_path in named.items():
                if left < right and (
                    left_path in right_path.parents or right_path in left_path.parents
                ):
                    raise ValueError(f"{left} and {right} roots must not overlap")
        return self


class ExecutableTask(BaseModel):
    task_id: str
    logical_task_ids: list[str]
    executor_id: str
    plugin_ids: list[str]
    stages: list[str]
    scope: ExecutionScope
    subject_id: str | None = None
    session_id: str | None = None
    acquisition_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    command: list[str]
    environment: dict[str, str] = Field(default_factory=dict)
    input_paths: list[str] = Field(default_factory=list)
    output_root: str
    work_dir: str
    resource_class: str
    cpus: int = Field(ge=1)
    memory_gb: int = Field(ge=1)
    gpus: int = Field(ge=0)
    walltime_minutes: int = Field(ge=1)
    required: bool = True
    artifacts: list[ArtifactRequirement] = Field(default_factory=list)
    fingerprint: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("command")
    @classmethod
    def command_is_argv(cls, value: list[str]) -> list[str]:
        if not value or any(not part for part in value):
            raise ValueError("command must be a non-empty argv list")
        return value


class ExecutionPlan(BaseModel):
    schema_version: str = "1.0"
    dataset_id: str
    generated_at: str
    catalog_digest: str
    layout: RunLayout
    tasks: list[ExecutableTask]
    required_executor_ids: list[str]
    optional_executor_ids: list[str]
    warnings: list[str] = Field(default_factory=list)
    run_policy: dict[str, Any] = Field(default_factory=dict)
    policy_digest: str | None = None
    backend_decisions: list[dict[str, Any]] = Field(default_factory=list)
    backend_qualification_tests: list[dict[str, Any]] = Field(default_factory=list)
    planned_smoke_tests: list[dict[str, Any]] = Field(default_factory=list)
    backend_evidence: BackendEvidenceBinding | None = None


class PreflightCheck(BaseModel):
    name: str
    status: str
    detail: str


class PreflightReport(BaseModel):
    ready: bool
    checks: list[PreflightCheck]
    missing_images: list[str] = Field(default_factory=list)
    missing_executables: list[str] = Field(default_factory=list)
    missing_runtime_assets: list[str] = Field(default_factory=list)
    missing_metadata: dict[str, list[str]] = Field(default_factory=dict)
