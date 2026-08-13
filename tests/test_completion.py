from neuroprep_agentic.completion import evaluate_completion
from neuroprep_agentic.identity import IdentityResolver
from neuroprep_agentic.ingestion import IngestionEngine
from neuroprep_agentic.models import (
    ArtifactRecord,
    CompletionContract,
    DatasetSpec,
    SourceRecord,
    TaskResult,
)
from neuroprep_agentic.planning import PipelinePlanner
from neuroprep_agentic.registry import builtin_registry


def graph_and_plan():
    registry = builtin_registry()
    graph = IngestionEngine(
        registry,
        IdentityResolver("a-strong-test-key"),
    ).build_graph(
        DatasetSpec(dataset_id="study"),
        [
            SourceRecord(
                dataset_id="study",
                source_path="/raw/t1.dcm",
                source_subject_id="p1",
                source_session_id="v1",
                metadata={"Modality": "MR", "SeriesDescription": "T1w MPRAGE"},
            )
        ],
    )
    return graph, PipelinePlanner(registry).plan(graph)


def test_completion_uses_contract_not_global_modality_assumptions():
    graph, plan = graph_and_plan()
    results = [
        TaskResult(task_id=task.task_id, status="complete", qc_passed=True)
        for task in plan.tasks
        if task.required
    ]
    artifact = ArtifactRecord(
        artifact_id="index",
        kind="patient_index",
        path="/derivatives/patient-index.jsonl",
        size=100,
        valid=True,
    )
    result = evaluate_completion(
        graph,
        plan,
        results,
        [artifact],
        CompletionContract(
            required_modalities={"structural_mri_t1w"},
            required_artifact_kinds={"patient_index"},
        ),
    )
    assert result.complete


def test_missing_required_modality_fails_explicitly():
    graph, plan = graph_and_plan()
    result = evaluate_completion(
        graph,
        plan,
        [],
        [],
        CompletionContract(required_modalities={"positron_emission_tomography"}),
    )
    assert not result.complete
    assert any("Missing required modalities" in reason for reason in result.reasons)


def test_optional_unacquired_modalities_do_not_fail():
    graph, plan = graph_and_plan()
    results = [
        TaskResult(task_id=task.task_id, status="complete", qc_passed=True)
        for task in plan.tasks
        if task.required
    ]
    result = evaluate_completion(
        graph,
        plan,
        results,
        [],
        CompletionContract(),
    )
    assert result.complete
