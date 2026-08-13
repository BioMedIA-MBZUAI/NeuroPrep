from __future__ import annotations

import json

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict

from .execution_models import ExecutionPlan
from .executors import builtin_executor_catalog
from .preflight import preflight_execution_plan
from .registry import builtin_registry
from .services import build_ingest_plan, load_manifest
from .settings import AgenticSettings, validated_cluster_path


class EmptyInput(BaseModel):
    model_config = ConfigDict(extra="ignore")


class CoverageInventoryTool(BaseTool):
    name: str = "NeuroPrep plugin coverage inventory"
    description: str = (
        "Return the registered dataset adapters, modality plugins, BIDS support levels, "
        "and processing coverage. This is read-only and dataset-independent."
    )
    args_schema: type[BaseModel] = EmptyInput

    def _run(self, **_ignored: object) -> str:
        return json.dumps(builtin_registry().inventory(), sort_keys=True)


class ManifestIngestPlanTool(BaseTool):
    name: str = "NeuroPrep manifest ingestion planner"
    description: str = (
        "Read the configured cluster manifest, build deterministic pseudonymous identities and "
        "longitudinal sessions, classify acquisitions, create BIDS-style mappings, quarantine "
        "ambiguities, and return a modality-aware preprocessing DAG. It does not modify data."
    )
    args_schema: type[BaseModel] = EmptyInput

    def _run(self, **_ignored: object) -> str:
        settings = AgenticSettings.from_env("ingest_plan")
        manifest = load_manifest(settings.manifest_path or "")
        result = build_ingest_plan(manifest, settings.identity_key or "")
        return json.dumps(result, sort_keys=True)


class ExecutionPlanReviewTool(BaseTool):
    name: str = "NeuroPrep deterministic execution plan reviewer"
    description: str = (
        "Read the configured fingerprinted execution plan and recompute its fail-closed "
        "preflight. Return exact task, executor, resource, dependency, and readiness evidence. "
        "This tool cannot submit, claim, retry, cancel, or execute jobs."
    )
    args_schema: type[BaseModel] = EmptyInput

    def _run(self, **_ignored: object) -> str:
        settings = AgenticSettings.from_env("execution_review")
        path = validated_cluster_path(settings.execution_plan_path or "", must_exist=True)
        plan = ExecutionPlan.model_validate_json(path.read_text(encoding="utf-8"))
        report = preflight_execution_plan(plan, builtin_executor_catalog())
        resources: dict[str, int] = {}
        for task in plan.tasks:
            resources[task.resource_class] = resources.get(task.resource_class, 0) + 1
        result = {
            "dataset_id": plan.dataset_id,
            "catalog_digest": plan.catalog_digest,
            "task_count": len(plan.tasks),
            "required_executor_ids": plan.required_executor_ids,
            "optional_executor_ids": plan.optional_executor_ids,
            "resource_counts": resources,
            "dependency_edges": sum(len(task.depends_on) for task in plan.tasks),
            "preflight": report.model_dump(mode="json"),
            "authority_note": (
                "This evidence is deterministic. Agent narrative cannot change the plan "
                "or authorize execution."
            ),
        }
        return json.dumps(result, sort_keys=True)


class MutationPolicyInput(BaseModel):
    action_type: str
    target_type: str
    target_id: str


class MutationPolicyTool(BaseTool):
    name: str = "NeuroPrep mutation policy"
    description: str = (
        "Evaluate a proposed action. The current foundation is read-only, so execution, deletion, "
        "identity linking, scheduler submission, and data mutation require a future bounded "
        "executor plus explicit policy. This tool never executes an action."
    )
    args_schema: type[BaseModel] = MutationPolicyInput

    def _run(self, action_type: str, target_type: str, target_id: str) -> str:
        payload = {
            "action_type": action_type,
            "target_type": target_type,
            "target_id": target_id,
            "allowed": False,
            "requires_approval": True,
            "reason": "No mutating executor is exposed by the foundation milestone.",
        }
        return json.dumps(payload, sort_keys=True)
