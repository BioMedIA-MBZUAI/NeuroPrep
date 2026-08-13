from __future__ import annotations

from collections import Counter

from .models import (
    ArtifactRecord,
    CanonicalAcquisitionGraph,
    CompletionContract,
    CompletionResult,
    PlanResult,
    TaskResult,
)


def evaluate_completion(
    graph: CanonicalAcquisitionGraph,
    plan: PlanResult,
    task_results: list[TaskResult],
    artifacts: list[ArtifactRecord],
    contract: CompletionContract,
) -> CompletionResult:
    reasons: list[str] = []
    task_statuses = {result.task_id: result for result in task_results}
    recognized_modalities = {
        acquisition.modality for acquisition in graph.acquisitions if acquisition.modality
    }
    missing_modalities = sorted(contract.required_modalities - recognized_modalities)
    if missing_modalities:
        reasons.append("Missing required modalities: " + ", ".join(missing_modalities))

    if graph.quarantined and not contract.allow_quarantine:
        reasons.append(f"{len(graph.quarantined)} acquisitions remain quarantined")
    if graph.unsupported and not contract.allow_unsupported:
        reasons.append(f"{len(graph.unsupported)} acquisitions remain unsupported")

    required_tasks = [task for task in plan.tasks if task.required]
    if contract.require_all_planned_tasks:
        incomplete = [
            task.task_id
            for task in required_tasks
            if task.task_id not in task_statuses or task_statuses[task.task_id].status != "complete"
        ]
        if incomplete:
            reasons.append(f"{len(incomplete)} required tasks are not complete")

    if contract.require_qc:
        qc_failures = [
            task.task_id
            for task in required_tasks
            if task.task_id in task_statuses and task_statuses[task.task_id].qc_passed is not True
        ]
        if qc_failures:
            reasons.append(f"{len(qc_failures)} required tasks lack passing QC")

    valid_artifact_kinds = {
        artifact.kind for artifact in artifacts if artifact.valid and artifact.size > 0
    }
    missing_artifacts = sorted(contract.required_artifact_kinds - valid_artifact_kinds)
    if missing_artifacts:
        reasons.append("Missing valid artifacts: " + ", ".join(missing_artifacts))

    status_counts = Counter(result.status for result in task_results)
    counts = {
        "subjects": len(graph.subjects),
        "acquisitions": len(graph.acquisitions),
        "quarantined": len(graph.quarantined),
        "unsupported": len(graph.unsupported),
        "planned_tasks": len(plan.tasks),
        **{f"task_{status}": count for status, count in sorted(status_counts.items())},
    }
    return CompletionResult(complete=not reasons, reasons=reasons, counts=counts)
