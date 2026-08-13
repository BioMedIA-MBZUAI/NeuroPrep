from datetime import datetime

from neuroprep_agentic.identity import IdentityResolver
from neuroprep_agentic.ingestion import IngestionEngine
from neuroprep_agentic.models import DatasetSpec, SourceRecord
from neuroprep_agentic.planning import PipelinePlanner
from neuroprep_agentic.registry import builtin_registry


def build(records):
    registry = builtin_registry()
    graph = IngestionEngine(
        registry,
        IdentityResolver("a-strong-test-key"),
    ).build_graph(DatasetSpec(dataset_id="study"), records)
    return graph, PipelinePlanner(registry).plan(graph)


def source(path, description, session="v1", date=None):
    return SourceRecord(
        dataset_id="study",
        source_path=path,
        source_subject_id="p1",
        source_session_id=session,
        acquisition_time=date,
        metadata={"Modality": "MR", "SeriesDescription": description},
    )


def test_dti_is_planned_as_dwi_derivative():
    _, plan = build([source("/raw/dwi.dcm", "DWI diffusion")])
    stages = [task.stage for task in plan.tasks]
    assert "dwi_preprocess" in stages
    assert "dti_fit" in stages
    assert not any(task.plugin_id == "dti" for task in plan.tasks)


def test_multimodal_session_gets_coregistration_task():
    records = [
        source("/raw/t1.dcm", "T1w MPRAGE"),
        source("/raw/bold.dcm", "resting state BOLD"),
    ]
    _, plan = build(records)
    task = next(task for task in plan.tasks if task.stage == "multimodal_coregistration")
    assert len(task.acquisition_ids) == 2
    assert task.depends_on


def test_two_structural_sessions_get_longitudinal_template():
    records = [
        source("/raw/v1_t1.dcm", "T1w MPRAGE", "v1", datetime(2025, 1, 1)),
        source("/raw/v2_t1.dcm", "T1w MPRAGE", "v2", datetime(2026, 1, 1)),
    ]
    _, plan = build(records)
    task = next(task for task in plan.tasks if task.stage == "longitudinal_subject_template")
    assert len(task.acquisition_ids) == 2
    assert task.resource.resource_class == "high_memory"


def test_missing_modalities_do_not_create_failure():
    _, plan = build([source("/raw/flair.dcm", "FLAIR")])
    assert not plan.issues
    assert plan.modality_counts == {"structural_mri_flair": 1}
