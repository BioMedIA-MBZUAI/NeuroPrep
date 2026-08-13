from __future__ import annotations

import json
from pathlib import Path

from neuroprep_agentic.compiler import (
    compile_execution_plan,
    executable_task_fingerprint,
    save_execution_plan,
)
from neuroprep_agentic.execution_models import RunLayout
from neuroprep_agentic.executors import builtin_executor_catalog
from neuroprep_agentic.models import (
    AcquisitionRecord,
    CanonicalAcquisitionGraph,
    ClassificationCandidate,
    ClassificationStatus,
    DatasetSpec,
    PlanResult,
    SessionRecord,
    SourceRecord,
    SubjectRecord,
    TaskSpec,
)
from neuroprep_agentic.registry import builtin_registry
from neuroprep_agentic.services import ManifestDocument, build_graph_and_plan


def _manifest(root: Path) -> ManifestDocument:
    t1 = root / "source" / "P001" / "V1" / "T1w.nii.gz"
    bold = root / "source" / "P001" / "V1" / "rest_bold.nii.gz"
    dwi = root / "source" / "P001" / "V1" / "diffusion_dwi.nii.gz"
    eeg = root / "source" / "P001" / "V1" / "rest_eeg.edf"
    t1.parent.mkdir(parents=True)
    t1.write_bytes(b"nifti-test")
    bold.write_bytes(b"nifti-test")
    dwi.write_bytes(b"nifti-test")
    eeg.write_bytes(b"edf-test")
    return ManifestDocument.model_validate(
        {
            "dataset": {
                "dataset_id": "compiler-test",
                "adapter_id": "generic",
                "required_modalities": ["structural_mri_t1w"],
            },
            "records": [
                {
                    "dataset_id": "compiler-test",
                    "source_path": str(t1),
                    "source_subject_id": "P001",
                    "source_session_id": "V1",
                    "file_format": "NIfTI",
                    "metadata": {
                        "Modality": "MR",
                        "SeriesDescription": "MPRAGE T1w",
                    },
                },
                {
                    "dataset_id": "compiler-test",
                    "source_path": str(bold),
                    "source_subject_id": "P001",
                    "source_session_id": "V1",
                    "file_format": "NIfTI",
                    "metadata": {
                        "Modality": "MR",
                        "SeriesDescription": "resting state BOLD fMRI",
                        "RepetitionTime": 2.0,
                        "TaskName": "rest",
                    },
                },
                {
                    "dataset_id": "compiler-test",
                    "source_path": str(dwi),
                    "source_subject_id": "P001",
                    "source_session_id": "V1",
                    "file_format": "NIfTI",
                    "metadata": {
                        "Modality": "MR",
                        "SeriesDescription": "diffusion DWI",
                        "PhaseEncodingDirection": "j-",
                        "TotalReadoutTime": 0.08,
                    },
                },
                {
                    "dataset_id": "compiler-test",
                    "source_path": str(eeg),
                    "source_subject_id": "P001",
                    "source_session_id": "V1",
                    "file_format": "EDF",
                    "metadata": {
                        "Modality": "EEG",
                        "RecordingType": "resting EEG",
                    },
                },
            ],
        }
    )


def _compile(root: Path):
    manifest = _manifest(root)
    graph, logical = build_graph_and_plan(manifest, "compiler-test-identity-key")
    layout = RunLayout(
        dataset_id="compiler-test",
        source_root=str(root / "source"),
        bids_root=str(root / "bids"),
        derivatives_root=str(root / "derivatives"),
        work_root=str(root / "work"),
        state_path=str(root / "state" / "tasks.sqlite"),
        freesurfer_license=str(root / "license.txt"),
    )
    return compile_execution_plan(
        graph,
        logical,
        layout,
        builtin_executor_catalog(),
    )


def test_compiler_emits_fingerprinted_dependency_dag(nfs_tmp: Path) -> None:
    plan = _compile(nfs_tmp)
    ids = {task.task_id for task in plan.tasks}
    assert len(ids) == len(plan.tasks)
    assert plan.required_executor_ids
    assert all(task.command for task in plan.tasks)
    assert all(set(task.depends_on) <= ids for task in plan.tasks)
    assert all(task.fingerprint == executable_task_fingerprint(task) for task in plan.tasks)

    bids_ids = {task.task_id for task in plan.tasks if task.executor_id == "neuroprep-bids-stage"}
    assert bids_ids
    bids_init = next(task for task in plan.tasks if task.executor_id == "neuroprep-bids-init")
    bids_validator = next(task for task in plan.tasks if task.executor_id == "bids-validator")
    assert all(
        bids_init.task_id in task.depends_on
        for task in plan.tasks
        if task.executor_id == "neuroprep-bids-stage"
    )
    assert set(bids_validator.depends_on) == bids_ids
    scientific = [
        task
        for task in plan.tasks
        if task.executor_id not in {"neuroprep-bids-init", "neuroprep-bids-stage", "bids-validator"}
    ]
    assert scientific
    assert all(bids_validator.task_id in task.depends_on for task in scientific)

    smriprep = next(task for task in plan.tasks if task.executor_id == "smriprep")
    deepprep = next(task for task in plan.tasks if task.executor_id == "deepprep")
    assert smriprep.metadata["primary_anatomical_bids_input"] in smriprep.metadata["bids_inputs"]
    assert smriprep.metadata["primary_anatomical_candidate_count"] == 1
    assert smriprep.task_id in deepprep.depends_on
    assert deepprep.metadata["primary_anatomical_bids_input"] in deepprep.metadata["bids_inputs"]
    assert deepprep.metadata["primary_anatomical_candidate_count"] == 1
    assert plan.tasks.index(smriprep) < plan.tasks.index(deepprep)
    for executor_id in ("fmriprep", "qsiprep"):
        task = next(task for task in plan.tasks if task.executor_id == executor_id)
        assert smriprep.task_id in task.depends_on
    xcpd = next(task for task in plan.tasks if task.executor_id == "xcp-d")
    fmriprep = next(task for task in plan.tasks if task.executor_id == "fmriprep")
    assert fmriprep.task_id in xcpd.depends_on


def test_saved_plan_and_task_contexts_round_trip(nfs_tmp: Path) -> None:
    plan = _compile(nfs_tmp)
    target = nfs_tmp / "run" / "execution-plan.json"
    save_execution_plan(plan, str(target))
    assert json.loads(target.read_text())["catalog_digest"] == plan.catalog_digest
    contexts = nfs_tmp / "work" / "contexts"
    assert len(list(contexts.glob("*.json"))) == len(plan.tasks)
    container_tasks = [task for task in plan.tasks if task.command[0] == "/usr/bin/singularity"]
    assert container_tasks
    for task in container_tasks:
        command = " ".join(task.command)
        assert f"{task.work_dir}/tmp:/tmp" in command
        assert f"{task.work_dir}/var-tmp:/var/tmp" in command
        assert f"--home {task.work_dir}/home" in command


def test_every_executor_compiles_in_one_dataset_agnostic_dag(nfs_tmp: Path) -> None:
    registry = builtin_registry()
    acquisitions = []
    logical_tasks = []
    for index, plugin in enumerate(registry.modalities.values(), start=1):
        extension = next((value for value in plugin.extensions if value), ".nii.gz")
        source = SourceRecord(
            dataset_id="all-modalities",
            source_path=str(nfs_tmp / "source" / f"{index:02d}-{plugin.plugin_id}{extension}"),
            source_subject_id="source-all",
            source_session_id="visit-1",
            metadata={"Modality": "TEST", "SeriesDescription": plugin.modality},
        )
        acquisition_id = f"acq-{index:02d}"
        bids = plugin.build_bids_mapping(
            source, {"subject": "all", "session": "01", "acquisition": f"{index:02d}"}
        )
        acquisitions.append(
            AcquisitionRecord(
                acquisition_id=acquisition_id,
                source=source,
                subject_id="sub-all",
                session_id="ses-01",
                status=ClassificationStatus.RECOGNIZED,
                candidates=[
                    ClassificationCandidate(
                        plugin_id=plugin.plugin_id,
                        modality=plugin.modality,
                        bids_datatype=plugin.bids_datatype,
                        bids_suffix=plugin.bids_suffix,
                        support_level=plugin.support_level,
                        score=1.0,
                        reasons=["test"],
                    )
                ],
                selected_plugin_id=plugin.plugin_id,
                modality=plugin.modality,
                bids=bids,
            )
        )
        logical_tasks.extend(plugin.plan_tasks(acquisition_id, "sub-all", "ses-01"))

    acquisition_ids = [item.acquisition_id for item in acquisitions]
    logical_tasks.extend(
        [
            TaskSpec(
                task_id="sub-all:ses-01:multimodal",
                plugin_id="core.multimodal",
                stage="multimodal_coregistration",
                subject_id="sub-all",
                session_id="ses-01",
                acquisition_ids=acquisition_ids,
            ),
            TaskSpec(
                task_id="sub-all:longitudinal",
                plugin_id="core.longitudinal",
                stage="longitudinal_subject_template",
                subject_id="sub-all",
                acquisition_ids=acquisition_ids,
            ),
        ]
    )
    graph = CanonicalAcquisitionGraph(
        dataset=DatasetSpec(dataset_id="all-modalities"),
        subjects=[
            SubjectRecord(
                subject_id="sub-all",
                dataset_id="all-modalities",
                source_subject_ids={"source-all"},
                sessions=[
                    SessionRecord(
                        session_id="ses-01",
                        source_session_ids={"visit-1"},
                        acquisitions=acquisitions,
                    )
                ],
            )
        ],
    )
    layout = RunLayout(
        dataset_id="all-modalities",
        source_root=str(nfs_tmp / "source"),
        bids_root=str(nfs_tmp / "bids"),
        derivatives_root=str(nfs_tmp / "derivatives"),
        work_root=str(nfs_tmp / "work-all"),
        state_path=str(nfs_tmp / "state-all" / "tasks.sqlite"),
        freesurfer_license=str(nfs_tmp / "license.txt"),
        execute_optional=True,
    )
    catalog = builtin_executor_catalog()
    plan = compile_execution_plan(
        graph,
        PlanResult(tasks=logical_tasks),
        layout,
        catalog,
    )
    assert {task.executor_id for task in plan.tasks} == {
        definition.executor_id for definition in catalog.definitions
    }
    assert all(task.command and task.artifacts for task in plan.tasks)
    assert any(task.executor_id == "fsl-mrs" for task in plan.tasks)
    assert any(task.executor_id == "neuroprep-ultrasound" for task in plan.tasks)
    assert any(task.executor_id == "neuroprep-angio" for task in plan.tasks)
