from __future__ import annotations

from collections import Counter

from .models import (
    CanonicalAcquisitionGraph,
    ClassificationStatus,
    PlanResult,
    ResourceRequest,
    Severity,
    TaskSpec,
    ValidationIssue,
)
from .registry import PluginRegistry

STRUCTURAL_PLUGINS = {"mri.t1w", "mri.t2w", "mri.flair", "mri.unit1"}


class PipelinePlanner:
    """Build a dependency-aware plan from recognized acquisitions only."""

    def __init__(self, registry: PluginRegistry) -> None:
        self.registry = registry

    def plan(self, graph: CanonicalAcquisitionGraph) -> PlanResult:
        tasks: list[TaskSpec] = []
        issues: list[ValidationIssue] = []
        modality_counts: Counter[str] = Counter()

        for acquisition in graph.acquisitions:
            if acquisition.status is ClassificationStatus.RECOGNIZED:
                plugin = self.registry.modality_plugin(acquisition.selected_plugin_id or "")
                modality_counts[plugin.modality] += 1
                tasks.extend(
                    plugin.plan_tasks(
                        acquisition.acquisition_id,
                        acquisition.subject_id,
                        acquisition.session_id,
                    )
                )
            elif acquisition.status is ClassificationStatus.AMBIGUOUS:
                issues.append(
                    ValidationIssue(
                        code="AMBIGUOUS_MODALITY",
                        severity=Severity.ERROR,
                        message="Acquisition requires classification review.",
                        acquisition_id=acquisition.acquisition_id,
                        path=acquisition.source.source_path,
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        code="UNSUPPORTED_MODALITY",
                        severity=Severity.WARNING,
                        message="Source is preserved but no preprocessing task was planned.",
                        acquisition_id=acquisition.acquisition_id,
                        path=acquisition.source.source_path,
                    )
                )

        for subject in graph.subjects:
            structural_sessions: dict[str, list[str]] = {}
            for session in subject.sessions:
                structural = [
                    acquisition
                    for acquisition in session.acquisitions
                    if acquisition.selected_plugin_id in STRUCTURAL_PLUGINS
                ]
                if structural:
                    structural_sessions[session.session_id] = [
                        acquisition.acquisition_id for acquisition in structural
                    ]
                recognized = [
                    acquisition
                    for acquisition in session.acquisitions
                    if acquisition.status is ClassificationStatus.RECOGNIZED
                ]
                if len({item.selected_plugin_id for item in recognized}) > 1:
                    task_id = f"{subject.subject_id}:{session.session_id}:multimodal_coregistration"
                    dependencies = [
                        task.task_id
                        for task in tasks
                        if task.subject_id == subject.subject_id
                        and task.session_id == session.session_id
                        and task.required
                    ]
                    tasks.append(
                        TaskSpec(
                            task_id=task_id,
                            plugin_id="core.multimodal",
                            stage="multimodal_coregistration",
                            subject_id=subject.subject_id,
                            session_id=session.session_id,
                            acquisition_ids=[item.acquisition_id for item in recognized],
                            depends_on=dependencies,
                            resource=ResourceRequest(cpus=8, memory_gb=24, walltime_minutes=180),
                        )
                    )

            if len(structural_sessions) >= 2:
                dependencies = [
                    task.task_id
                    for task in tasks
                    if task.subject_id == subject.subject_id
                    and task.plugin_id in STRUCTURAL_PLUGINS
                    and task.required
                ]
                tasks.append(
                    TaskSpec(
                        task_id=f"{subject.subject_id}:longitudinal_subject_template",
                        plugin_id="core.longitudinal",
                        stage="longitudinal_subject_template",
                        subject_id=subject.subject_id,
                        acquisition_ids=[
                            item for values in structural_sessions.values() for item in values
                        ],
                        depends_on=dependencies,
                        resource=ResourceRequest(
                            resource_class="high_memory",
                            cpus=16,
                            memory_gb=64,
                            walltime_minutes=480,
                        ),
                    )
                )

        return PlanResult(
            tasks=_deduplicate(tasks),
            issues=issues,
            modality_counts=dict(modality_counts),
        )


def _deduplicate(tasks: list[TaskSpec]) -> list[TaskSpec]:
    return list({task.task_id: task for task in tasks}.values())
