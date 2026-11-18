from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crewai.flow import Flow, listen, router, start
from dotenv import load_dotenv
from pydantic import BaseModel

from . import _PROJECT_ROOT
from .crews.control import kickoff_control_crew
from .execution_models import ExecutionPlan
from .executors import builtin_executor_catalog
from .preflight import preflight_execution_plan
from .registry import builtin_registry
from .services import build_ingest_plan, load_manifest
from .settings import AgenticSettings, WorkflowName, validated_cluster_path

PROJECT_ROOT = _PROJECT_ROOT
load_dotenv(PROJECT_ROOT / ".env")


class ControlState(BaseModel):
    workflow: WorkflowName = "coverage"
    request: str = "Inventory dataset and modality coverage."
    report: str = ""
    authoritative_artifact: str = ""


def authoritative_evidence(
    workflow: WorkflowName,
    settings: AgenticSettings,
) -> dict[str, Any]:
    """Recompute control evidence without trusting an LLM transcription."""

    if workflow == "coverage":
        return builtin_registry().inventory()
    if workflow == "ingest_plan":
        manifest = load_manifest(settings.manifest_path or "")
        return build_ingest_plan(manifest, settings.identity_key or "")
    path = validated_cluster_path(settings.execution_plan_path or "", must_exist=True)
    plan = ExecutionPlan.model_validate_json(path.read_text(encoding="utf-8"))
    report = preflight_execution_plan(plan, builtin_executor_catalog())
    return {
        "dataset_id": plan.dataset_id,
        "catalog_digest": plan.catalog_digest,
        "task_count": len(plan.tasks),
        "required_executor_ids": plan.required_executor_ids,
        "optional_executor_ids": plan.optional_executor_ids,
        "preflight": report.model_dump(mode="json"),
    }


def write_control_artifact(
    workflow: WorkflowName,
    evidence: dict[str, Any],
    narrative: str,
) -> Path:
    """Persist authoritative evidence next to explicitly advisory agent prose."""

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_label = os.getenv("SLURM_JOB_ID") or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = output_dir / f"control-{workflow}-{run_label}.json"
    temporary = destination.with_suffix(".json.tmp")
    payload = {
        "schema_version": "1.0",
        "workflow": workflow,
        "narrative_authoritative": False,
        "authority_note": (
            "Only authoritative_evidence may drive mappings, execution, validation, "
            "or completion. Agent narrative is advisory."
        ),
        "authoritative_evidence": evidence,
        "agent_narrative": narrative,
    }
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def validate_control_payload(
    payload: dict[str, Any],
    workflow: WorkflowName,
    expected_evidence: dict[str, Any],
) -> None:
    """Reject a control artifact whose authoritative section was altered."""

    if payload.get("workflow") != workflow:
        raise ValueError("Control artifact workflow does not match the requested workflow")
    if payload.get("narrative_authoritative") is not False:
        raise ValueError("Agent narrative must never be marked authoritative")
    if payload.get("authoritative_evidence") != expected_evidence:
        raise ValueError("Authoritative evidence does not match deterministic recomputation")


def validate_control_artifact(path: str, workflow: WorkflowName) -> Path:
    settings = AgenticSettings.from_env(workflow)
    candidate = Path(path)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    validate_control_payload(payload, workflow, authoritative_evidence(workflow, settings))
    return candidate


class NeuroPrepControlFlow(Flow[ControlState]):
    """Typed modality- and dataset-independent CrewAI Flow."""

    @start()
    def prepare(self) -> str:
        settings = AgenticSettings.from_env(self.state.workflow)
        self.state.workflow = settings.workflow
        return self.state.workflow

    @router(prepare)
    def route(self) -> str:
        return self.state.workflow

    def _run(self, workflow: WorkflowName) -> str:
        settings = AgenticSettings.from_env(workflow)
        result = kickoff_control_crew(
            workflow,
            {
                "request": self.state.request,
                "manifest_path": settings.manifest_path or "not supplied",
            },
        )
        self.state.report = result.raw
        evidence = authoritative_evidence(workflow, settings)
        artifact = write_control_artifact(workflow, evidence, self.state.report)
        self.state.authoritative_artifact = str(artifact)
        print(f"Authoritative control artifact: {artifact}")
        return self.state.report

    @listen("coverage")
    def run_coverage(self) -> str:
        return self._run("coverage")

    @listen("ingest_plan")
    def run_ingest_plan(self) -> str:
        return self._run("ingest_plan")

    @listen("execution_review")
    def run_execution_review(self) -> str:
        return self._run("execution_review")


def run_control_flow(workflow: WorkflowName, request: str) -> NeuroPrepControlFlow:
    (PROJECT_ROOT / "output").mkdir(parents=True, exist_ok=True)
    flow = NeuroPrepControlFlow()
    flow.kickoff(inputs={"workflow": workflow, "request": request})
    return flow


def kickoff() -> None:
    settings = AgenticSettings.from_env()
    request = os.getenv(
        "NEUROPREP_REQUEST",
        "Inventory the current dataset and modality coverage.",
    )
    run_control_flow(settings.workflow, request)


def plot() -> None:
    NeuroPrepControlFlow().plot(str(PROJECT_ROOT / "neuroprep_control_flow"))
