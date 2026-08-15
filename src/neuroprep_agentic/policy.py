from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from contextlib import suppress
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProcessingObjective(StrEnum):
    STANDARD = "standard"
    HIGH_QUALITY = "high_quality"
    FAST = "fast"
    LOW_MEMORY = "low_memory"
    EXPLORATORY = "exploratory"


KNOWN_PACKAGES = frozenset(
    {
        "ants",
        "aslprep",
        "basil",
        "deepprep",
        "dipy",
        "eeglab",
        "exploreasl",
        "fastsurfer",
        "fmriprep",
        "freesurfer",
        "fsl",
        "fsl-mrs",
        "lcmodel",
        "mne",
        "mrtrix3",
        "petprep",
        "petpvc",
        "qsiprep",
        "qsirecon",
        "qsmxt",
        "smriprep",
        "spm",
        "synthseg",
        "totalsegmentator",
        "xcp-d",
    }
)


class HardwareProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hostname: str = "unknown"
    cpus: int = Field(default=1, ge=1)
    memory_gb: int = Field(default=4, ge=1)
    gpus: int = Field(default=0, ge=0)
    gpu_memory_gb: int = Field(default=0, ge=0)
    gpu_models: list[str] = Field(default_factory=list)
    cuda_version: str | None = None
    container_runtime: Literal["singularity", "apptainer", "none"] = "none"


class RunPolicy(BaseModel):
    """Validated scientific intent compiled from a prompt.

    No field accepts a command, executable, image, path, or scheduler directive.
    The prompt can influence goals and preferences but cannot bypass the catalog,
    metadata gates, validation, or completion contracts.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    source: Literal["default", "prompt", "file"] = "default"
    source_prompt: str = "Process every detected modality with strict validation."
    objective: ProcessingObjective = ProcessingObjective.STANDARD
    requested_outputs: list[
        Literal[
            "bids",
            "preprocessed_images",
            "segmentations",
            "quantitative_maps",
            "timeseries",
            "connectomes",
            "tractography",
            "qc_reports",
            "model_ready_index",
            "tensor_index",
        ]
    ] = Field(
        default_factory=lambda: [
            "bids",
            "preprocessed_images",
            "segmentations",
            "quantitative_maps",
            "qc_reports",
            "model_ready_index",
            "tensor_index",
        ]
    )
    preferred_packages: list[str] = Field(default_factory=list)
    require_gpu_when_supported: bool = True
    allow_experimental_backends: bool = False
    max_subject_walltime_hours: int | None = Field(default=None, ge=1, le=336)
    notes: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("preferred_packages")
    @classmethod
    def packages_are_allowlisted(cls, value: list[str]) -> list[str]:
        normalized = [item.strip().lower() for item in value]
        unknown = sorted(set(normalized) - KNOWN_PACKAGES)
        if unknown:
            raise ValueError("unknown package preferences: " + ", ".join(unknown))
        return list(dict.fromkeys(normalized))

    @field_validator("source_prompt")
    @classmethod
    def prompt_is_bounded(cls, value: str) -> str:
        if not value.strip() or len(value) > 4000:
            raise ValueError("source prompt must contain 1-4000 characters")
        return value.strip()

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class BackendCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability: str
    package_id: str
    executor_id: str | None = None
    integration_state: Literal["integrated", "candidate"]
    objective_tags: set[ProcessingObjective] = Field(default_factory=set)
    min_cpus: int = 1
    min_memory_gb: int = 4
    min_gpus: int = 0
    min_gpu_memory_gb: int = 0
    smoke_description: str
    expected_evidence: list[str]


class SmokeTestSpec(BaseModel):
    test_id: str
    capability: str
    package_id: str
    executor_id: str
    level: Literal["availability", "representative_subject"]
    description: str
    expected_evidence: list[str]
    bounded_minutes: int = Field(ge=1, le=720)


class BackendDecision(BaseModel):
    capability: str
    selected_package: str | None
    selected_executor_id: str | None
    selection_reasons: list[str]
    rejected_candidates: dict[str, str]
    requires_smoke_test: bool


class BackendPlan(BaseModel):
    schema_version: str = "1.0"
    policy_digest: str
    hardware: HardwareProfile
    decisions: list[BackendDecision]
    smoke_tests: list[SmokeTestSpec]
    ready: bool


def detect_hardware_profile() -> HardwareProfile:
    memory_gb = 4
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        memory_gb = max(1, int(page_size * pages / 1024**3))
    except (ValueError, OSError, AttributeError):
        pass
    gpu_models: list[str] = []
    gpu_memory = 0
    cuda_version: str | None = None
    if shutil.which("nvidia-smi"):
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        for line in completed.stdout.splitlines():
            parts = [item.strip() for item in line.split(",")]
            if len(parts) >= 2:
                gpu_models.append(parts[0])
                with suppress(ValueError):
                    gpu_memory = max(gpu_memory, int(float(parts[1]) / 1024))
        version = subprocess.run(
            ["nvidia-smi"], check=False, capture_output=True, text=True, timeout=20
        ).stdout
        marker = "CUDA Version:"
        if marker in version:
            cuda_version = version.split(marker, 1)[1].split()[0]
    allocated_cpus = os.getenv("SLURM_CPUS_PER_TASK") or os.getenv("SLURM_JOB_CPUS_PER_NODE")
    cpu_count = os.cpu_count() or 1
    if allocated_cpus:
        match = re.search(r"\d+", allocated_cpus)
        if match:
            cpu_count = max(1, int(match.group()))
    allocated_memory = os.getenv("SLURM_MEM_PER_NODE")
    if allocated_memory:
        match = re.search(r"\d+", allocated_memory)
        if match:
            memory_gb = max(1, int(match.group()) // 1024)
    allocated_gpu_value = os.getenv("SLURM_GPUS_ON_NODE") or os.getenv("SLURM_GPUS")
    if allocated_gpu_value:
        match = re.search(r"\d+", allocated_gpu_value)
        if match:
            allocated_gpus = int(match.group())
            gpu_models = gpu_models[:allocated_gpus]
    runtime: Literal["singularity", "apptainer", "none"] = "none"
    if shutil.which("singularity"):
        runtime = "singularity"
    elif shutil.which("apptainer"):
        runtime = "apptainer"
    return HardwareProfile(
        hostname=os.uname().nodename,
        cpus=cpu_count,
        memory_gb=memory_gb,
        gpus=len(gpu_models),
        gpu_memory_gb=gpu_memory,
        gpu_models=gpu_models,
        cuda_version=cuda_version,
        container_runtime=runtime,
    )


def builtin_backend_portfolio() -> tuple[BackendCandidate, ...]:
    def integrated(
        capability: str,
        package: str,
        executor: str,
        *,
        gpu: int = 0,
        gpu_memory: int = 0,
        tags: tuple[ProcessingObjective, ...] = (ProcessingObjective.STANDARD,),
    ) -> BackendCandidate:
        return BackendCandidate(
            capability=capability,
            package_id=package,
            executor_id=executor,
            integration_state="integrated",
            objective_tags=set(tags),
            min_cpus=4,
            min_memory_gb=16,
            min_gpus=gpu,
            min_gpu_memory_gb=gpu_memory,
            smoke_description=(
                "Run the pinned image/tool on one representative subject, validate declared "
                "artifacts and QC, then compare runtime and peak resources."
            ),
            expected_evidence=["return_code=0", "artifact_contract=pass", "qc_contract=pass"],
        )

    def candidate(capability: str, package: str, *, gpu: int = 0) -> BackendCandidate:
        return BackendCandidate(
            capability=capability,
            package_id=package,
            integration_state="candidate",
            min_cpus=4,
            min_memory_gb=16,
            min_gpus=gpu,
            smoke_description=(
                "Adapter, pinned environment, artifact contract, and reference-data "
                "qualification are required."
            ),
            expected_evidence=["adapter_integrated", "version_pinned", "reference_smoke_passed"],
        )

    return (
        integrated("structural_preprocess", "smriprep", "smriprep"),
        candidate("structural_preprocess", "freesurfer"),
        candidate("structural_preprocess", "synthseg", gpu=1),
        integrated("structural_segment", "deepprep", "deepprep", gpu=1, gpu_memory=8),
        candidate("structural_segment", "fastsurfer", gpu=1),
        candidate("structural_segment", "synthseg", gpu=1),
        integrated("bold_preprocess", "fmriprep", "fmriprep"),
        candidate("bold_preprocess", "deepprep", gpu=1),
        integrated("bold_postprocess", "xcp-d", "xcp-d"),
        integrated("dwi_preprocess", "qsiprep", "qsiprep", gpu=1, gpu_memory=8),
        candidate("dwi_preprocess", "mrtrix3"),
        candidate("dwi_preprocess", "fsl", gpu=1),
        integrated("diffusion_model", "qsirecon", "qsirecon", gpu=1, gpu_memory=8),
        candidate("diffusion_model", "mrtrix3"),
        candidate("diffusion_model", "dipy"),
        integrated("asl", "aslprep", "aslprep"),
        candidate("asl", "basil"),
        candidate("asl", "exploreasl"),
        integrated("pet", "petprep", "petprep"),
        candidate("pet", "petpvc"),
        candidate("pet", "spm"),
        integrated("swi_qsm", "qsmxt", "neuroprep-susceptibility", gpu=1),
        integrated("ct", "ants", "neuroprep-ct"),
        candidate("ct", "totalsegmentator", gpu=1),
        integrated("eeg", "mne", "neuroprep-eeg"),
        candidate("eeg", "eeglab"),
        integrated("mrs", "fsl-mrs", "fsl-mrs"),
        candidate("mrs", "lcmodel"),
    )


def plan_backends(
    policy: RunPolicy,
    hardware: HardwareProfile,
    evidence: dict[str, str] | None = None,
    *,
    required_executor_ids: set[str] | frozenset[str] | None = None,
) -> BackendPlan:
    observed = evidence or {}
    portfolio = builtin_backend_portfolio()
    if required_executor_ids is not None:
        active_capabilities = {
            item.capability
            for item in portfolio
            if item.executor_id is not None and item.executor_id in required_executor_ids
        }
        portfolio = tuple(
            item for item in portfolio if item.capability in active_capabilities
        )
    grouped: dict[str, list[BackendCandidate]] = {}
    for item in portfolio:
        grouped.setdefault(item.capability, []).append(item)
    decisions: list[BackendDecision] = []
    smoke_tests: list[SmokeTestSpec] = []
    for capability in sorted(grouped):
        scored: list[tuple[int, BackendCandidate, list[str]]] = []
        rejected: dict[str, str] = {}
        for item in grouped[capability]:
            if item.integration_state != "integrated":
                rejected[item.package_id] = "adapter_not_integrated_or_qualified"
                continue
            if item.min_gpus > hardware.gpus:
                rejected[item.package_id] = "insufficient_gpus"
                continue
            if item.min_gpu_memory_gb > hardware.gpu_memory_gb:
                rejected[item.package_id] = "insufficient_gpu_memory"
                continue
            if item.min_cpus > hardware.cpus or item.min_memory_gb > hardware.memory_gb:
                rejected[item.package_id] = "insufficient_cpu_or_memory"
                continue
            test_id = f"smoke--{capability}--{item.package_id}"
            if observed.get(test_id) == "failed":
                rejected[item.package_id] = "smoke_test_failed"
                continue
            score = 100
            reasons = ["integrated deterministic executor", "hardware compatible"]
            if item.package_id in policy.preferred_packages:
                score += 30
                reasons.append("prompt preference")
            if policy.objective in item.objective_tags:
                score += 10
                reasons.append(f"matches {policy.objective.value} objective")
            if observed.get(test_id) == "passed":
                score += 40
                reasons.append("representative smoke evidence passed")
            scored.append((score, item, reasons))
        scored.sort(key=lambda value: (-value[0], value[1].package_id))
        # Qualify every eligible challenger, not only the provisional winner.
        # The execution compiler still fingerprints only the selected backend
        # into a production DAG; these specifications drive separate bounded
        # representative-subject qualification runs.
        for _candidate_score, candidate_item, _candidate_reasons in scored:
            test_id = f"smoke--{capability}--{candidate_item.package_id}"
            if observed.get(test_id) == "passed" or candidate_item.executor_id is None:
                continue
            smoke_tests.append(
                SmokeTestSpec(
                    test_id=test_id,
                    capability=capability,
                    package_id=candidate_item.package_id,
                    executor_id=candidate_item.executor_id,
                    level="representative_subject",
                    description=candidate_item.smoke_description,
                    expected_evidence=candidate_item.expected_evidence,
                    bounded_minutes=720,
                )
            )
        selected = scored[0] if scored else None
        if selected is None:
            decisions.append(
                BackendDecision(
                    capability=capability,
                    selected_package=None,
                    selected_executor_id=None,
                    selection_reasons=["no integrated hardware-compatible backend"],
                    rejected_candidates=rejected,
                    requires_smoke_test=False,
                )
            )
            continue
        _score, item, reasons = selected
        test_id = f"smoke--{capability}--{item.package_id}"
        requires_smoke = observed.get(test_id) != "passed"
        decisions.append(
            BackendDecision(
                capability=capability,
                selected_package=item.package_id,
                selected_executor_id=item.executor_id,
                selection_reasons=reasons,
                rejected_candidates=rejected,
                requires_smoke_test=requires_smoke,
            )
        )
    return BackendPlan(
        policy_digest=policy.digest(),
        hardware=hardware,
        decisions=decisions,
        smoke_tests=smoke_tests,
        ready=all(item.selected_executor_id is not None for item in decisions),
    )


def compile_prompt_policy(request: str, hardware: HardwareProfile) -> RunPolicy:
    """Use the local Ollama-backed CrewAI orchestrator to produce typed intent."""

    from crewai import LLM, Agent, Crew, Process, Task

    from .settings import AgenticSettings

    settings = AgenticSettings.from_env("coverage")
    llm = LLM(
        model=settings.manager_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        timeout=300.0,
        max_retries=2,
    )
    agent = Agent(
        role="Central NeuroPrep Policy Compiler",
        goal="Translate user scientific intent into the constrained RunPolicy schema.",
        backstory=(
            "You cannot emit shell commands, paths, images, metadata values, modality labels, "
            "or scheduler instructions. You only choose among schema fields and allowlisted "
            "packages."
        ),
        llm=llm,
        function_calling_llm=llm,
        allow_delegation=False,
        verbose=True,
        max_iter=3,
    )
    task = Task(
        description=(
            "Compile this request into RunPolicy: {request}. Hardware evidence: {hardware}. "
            "Process all detected modalities unless the schema says otherwise. Unknown package "
            "names must not be added. Never invent acquisition metadata or commands."
        ),
        expected_output="A valid RunPolicy object with only the declared fields.",
        agent=agent,
        output_pydantic=RunPolicy,
    )
    result = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True).kickoff(
        inputs={
            "request": request,
            "hardware": json.dumps(hardware.model_dump(mode="json"), sort_keys=True),
        }
    )
    if result.pydantic is not None:
        proposed = RunPolicy.model_validate(result.pydantic)
    else:
        proposed = RunPolicy.model_validate_json(result.raw)
    return proposed.model_copy(update={"source": "prompt", "source_prompt": request.strip()})


def load_policy(path: str) -> RunPolicy:
    from .settings import validated_cluster_path

    candidate = validated_cluster_path(path, must_exist=True)
    return RunPolicy.model_validate_json(candidate.read_text(encoding="utf-8"))


def load_hardware_profile(path: str | None) -> HardwareProfile:
    if path is None:
        return detect_hardware_profile()
    from .settings import validated_cluster_path

    candidate = validated_cluster_path(path, must_exist=True)
    return HardwareProfile.model_validate_json(candidate.read_text(encoding="utf-8"))
