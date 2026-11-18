from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    pet = root / "source" / "P001" / "V1" / "amyloid_pet.nii.gz"
    eeg = root / "source" / "P001" / "V1" / "rest_eeg.edf"
    t1.parent.mkdir(parents=True)
    t1.write_bytes(b"nifti-test")
    bold.write_bytes(b"nifti-test")
    dwi.write_bytes(b"nifti-test")
    pet.write_bytes(b"nifti-test")
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
                    "source_path": str(pet),
                    "source_subject_id": "P001",
                    "source_session_id": "V1",
                    "file_format": "NIfTI",
                    "metadata": {
                        "Modality": "PT",
                        "SeriesDescription": "amyloid PET",
                        "TracerName": "florbetapir",
                        "Units": "Bq/mL",
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


def _compile(root: Path, *, dwi_bvals: str | None = None):
    manifest = _manifest(root)
    if dwi_bvals is not None:
        dwi = root / "source" / "P001" / "V1" / "diffusion_dwi.nii.gz"
        dwi.with_name("diffusion_dwi.bval").write_text(dwi_bvals)
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


def test_qsiprep_selects_shoreline_for_dsi_like_sampling(nfs_tmp: Path) -> None:
    plan = _compile(
        nfs_tmp,
        dwi_bvals="0 50 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300\n",
    )
    task = next(item for item in plan.tasks if item.executor_id == "qsiprep")
    assert task.command[-2:] == ["--hmc-model", "3dSHORE"]
    assert "--eddy-config" not in task.command
    assert task.metadata["dwi_sampling"]["selected_hmc_model"] == "3dSHORE"
    assert task.metadata["dwi_sampling"]["series_evidence"][0]["dsi_like"] is True


def test_qsiprep_retains_eddy_for_shelled_sampling(nfs_tmp: Path) -> None:
    plan = _compile(nfs_tmp, dwi_bvals="0 1000 1000 2000 2000\n")
    task = next(item for item in plan.tasks if item.executor_id == "qsiprep")
    assert "--hmc-model" not in task.command
    assert "--eddy-config" in task.command
    assert task.metadata["dwi_sampling"]["selected_hmc_model"] == "eddy"


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
    petprep = next(task for task in plan.tasks if task.executor_id == "petprep")
    assert smriprep.task_id in petprep.depends_on
    assert deepprep.task_id in petprep.depends_on
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
        definition.executor_id for definition in catalog.definitions if not definition.alternative
    }
    assert all(task.command and task.artifacts for task in plan.tasks)
    compared = compile_execution_plan(
        graph,
        PlanResult(tasks=logical_tasks),
        layout,
        catalog,
        qualification_candidates={
            "deepprep", "fastsurfer", "synthseg", "aslprep", "basil",
            "neuroprep-spatial", "totalsegmentator", "qsiprep", "mrtrix3",
        },
    )
    assert {task.executor_id for task in compared.tasks} == {
        definition.executor_id for definition in catalog.definitions
    }
    spatial = next(t for t in compared.tasks if t.executor_id == "neuroprep-spatial")
    assert spatial.required  # also does non-compared FLAIR/T2w/CT preprocessing
    totalseg = next(t for t in compared.tasks if t.executor_id == "totalsegmentator")
    assert not totalseg.required and totalseg.metadata["qualification_candidate"] is True
    image = totalseg.command[totalseg.command.index("-i") + 1]
    assert image.startswith("/source-data/sub-all/") and image.endswith(".nii.gz")
    assert totalseg.environment == {"TOTALSEG_HOME_DIR": "/totalseg-home"}
    basil = next(t for t in compared.tasks if t.executor_id == "basil")
    assert basil.command[basil.command.index("--image") + 1].endswith("fsl-6.0.7.18.sif")
    assert any(task.executor_id == "fsl-mrs" for task in plan.tasks)
    assert any(task.executor_id == "neuroprep-ultrasound" for task in plan.tasks)
    assert any(task.executor_id == "neuroprep-angio" for task in plan.tasks)


def test_container_commands_use_the_configured_runtime(monkeypatch, nfs_tmp: Path) -> None:
    monkeypatch.setenv("NEUROPREP_CONTAINER_RUNTIME", "/opt/apptainer/bin/apptainer")
    plan = _compile(nfs_tmp)
    container_tasks = [t for t in plan.tasks if t.executor_id in {"smriprep", "fmriprep"}]
    assert container_tasks
    assert {t.command[0] for t in container_tasks} == {"/opt/apptainer/bin/apptainer"}
    assert not any("/usr/bin/singularity" in part for t in plan.tasks for part in t.command)


def test_backend_decision_routes_and_pilot_compares_alternatives(nfs_tmp: Path) -> None:
    graph, logical = build_graph_and_plan(_manifest(nfs_tmp), "compiler-test-identity-key")
    layout = RunLayout(
        dataset_id="compiler-test",
        source_root=str(nfs_tmp / "source"),
        bids_root=str(nfs_tmp / "bids"),
        derivatives_root=str(nfs_tmp / "derivatives"),
        work_root=str(nfs_tmp / "work"),
        state_path=str(nfs_tmp / "state" / "tasks.sqlite"),
        freesurfer_license=str(nfs_tmp / "license.txt"),
    )
    catalog = builtin_executor_catalog()
    default = compile_execution_plan(graph, logical, layout, catalog)
    segmenters = {t.executor_id for t in default.tasks if "anat_segment" in t.stages}
    assert segmenters == {"deepprep"}

    decision = [{"capability": "structural_segment", "selected_executor_id": "fastsurfer"}]
    with pytest.raises(ValueError, match="requires a planned deepprep"):
        compile_execution_plan(graph, logical, layout, catalog, backend_decisions=decision)
    logical = PlanResult(tasks=[t for t in logical.tasks if t.plugin_id == "mri.t1w"])
    routed = compile_execution_plan(graph, logical, layout, catalog, backend_decisions=decision)
    task = next(t for t in routed.tasks if "anat_segment" in t.stages)
    assert task.executor_id == "fastsurfer" and task.required
    assert "deepprep" not in {t.executor_id for t in routed.tasks}
    t1 = task.command[task.command.index("--t1") + 1]
    assert t1.startswith("/source-data/sub-") and t1.endswith("_T1w.nii.gz")

    pilot = compile_execution_plan(
        graph, logical, layout, catalog, qualification_candidates={"deepprep", "fastsurfer"}
    )
    compared = [t for t in pilot.tasks if "anat_segment" in t.stages]
    assert {t.executor_id for t in compared} == {"deepprep", "fastsurfer"}
    assert all(not t.required and pilot.runs(t) for t in compared)
    assert all(t.metadata["qualification_candidate"] is True for t in compared)
    smriprep = next(t for t in pilot.tasks if t.executor_id == "smriprep")
    assert smriprep.required and "qualification_candidate" not in smriprep.metadata


def test_prefetch_covers_backends_a_pilot_will_compare(nfs_tmp: Path) -> None:
    from neuroprep_agentic.images import required_images

    plan = _compile(nfs_tmp).model_copy(
        update={
            "backend_qualification_tests": [
                {"test_id": "smoke--structural_segment--fastsurfer", "executor_id": "fastsurfer"}
            ]
        }
    )
    specs = required_images(plan, builtin_executor_catalog())
    assert "fastsurfer" in {spec.executor_id for spec in specs}


def test_selected_mrtrix3_takes_over_the_tensor_fit(nfs_tmp: Path) -> None:
    graph, logical = build_graph_and_plan(_manifest(nfs_tmp), "compiler-test-identity-key")
    layout = RunLayout(
        dataset_id="compiler-test",
        source_root=str(nfs_tmp / "source"),
        bids_root=str(nfs_tmp / "bids"),
        derivatives_root=str(nfs_tmp / "derivatives"),
        work_root=str(nfs_tmp / "work"),
        state_path=str(nfs_tmp / "state" / "tasks.sqlite"),
        freesurfer_license=str(nfs_tmp / "license.txt"),
    )
    catalog = builtin_executor_catalog()
    default = compile_execution_plan(graph, logical, layout, catalog)
    xcpd = next(t for t in default.tasks if t.executor_id == "xcp-d")
    assert xcpd.metadata["consumes_from"] == "fmriprep" and "/out/fmriprep" in xcpd.command
    fmriprep = next(t for t in default.tasks if t.executor_id == "fmriprep")
    assert fmriprep.task_id in xcpd.depends_on

    decisions = [
        {"capability": "dwi_preprocess", "selected_executor_id": "mrtrix3"},
        {"capability": "diffusion_model", "selected_executor_id": "qsirecon"},
    ]
    routed = compile_execution_plan(graph, logical, layout, catalog, backend_decisions=decisions)
    mrtrix3 = next(t for t in routed.tasks if t.executor_id == "mrtrix3")
    assert set(mrtrix3.stages) == {"dwi_preprocess", "dti_fit"} and mrtrix3.required
    assert not {"qsiprep", "qsirecon"} & {t.executor_id for t in routed.tasks}
    assert any("dropped optional qsirecon" in warning for warning in routed.warnings)
