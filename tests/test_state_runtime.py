from __future__ import annotations

import json
from pathlib import Path

from neuroprep_agentic.compiler import (
    compile_execution_plan,
    executable_task_fingerprint,
    save_execution_plan,
)
from neuroprep_agentic.execution_models import ExecutableTask, ExecutionScope, RunLayout
from neuroprep_agentic.executors import builtin_executor_catalog
from neuroprep_agentic.release import finalize_execution_run
from neuroprep_agentic.runtime import (
    _prepare_deepprep_bids_view,
    _prepare_primary_anatomical_bids_view,
    _task_environment,
    run_worker,
)
from neuroprep_agentic.services import ManifestDocument, build_graph_and_plan
from neuroprep_agentic.state import RunState


def _bids_only_plan(root: Path):
    source = root / "source" / "subject01" / "T1w.nii.gz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic-nifti")
    source.with_name("T1w.json").write_text(
        json.dumps(
            {
                "Modality": "MR",
                "SeriesDescription": "MPRAGE T1w",
                "PatientName": "must-not-leak",
                "StudyDate": "20200101",
                "PatientPosition": "HFS",
                "Nested": {"PatientID": "must-not-leak"},
            }
        )
    )
    manifest = ManifestDocument.model_validate(
        {
            "dataset": {"dataset_id": "runtime-test"},
            "records": [
                {
                    "dataset_id": "runtime-test",
                    "source_path": str(source),
                    "source_subject_id": "subject01",
                    "source_session_id": "baseline",
                    "file_format": "NIfTI",
                    "metadata": {
                        "Modality": "MR",
                        "SeriesDescription": "MPRAGE T1w",
                    },
                }
            ],
        }
    )
    graph, logical = build_graph_and_plan(manifest, "runtime-test-identity-key")
    layout = RunLayout(
        dataset_id="runtime-test",
        source_root=str(root / "source"),
        bids_root=str(root / "bids"),
        derivatives_root=str(root / "derivatives"),
        work_root=str(root / "work"),
        state_path=str(root / "state" / "tasks.sqlite"),
        freesurfer_license=str(root / "license.txt"),
    )
    plan = compile_execution_plan(
        graph,
        logical,
        layout,
        builtin_executor_catalog(),
    )
    bids_task = next(task for task in plan.tasks if task.executor_id == "neuroprep-bids-stage")
    bids_task = bids_task.model_copy(update={"depends_on": [], "fingerprint": "pending"})
    bids_task = bids_task.model_copy(update={"fingerprint": executable_task_fingerprint(bids_task)})
    return plan.model_copy(
        update={
            "tasks": [bids_task],
            "required_executor_ids": ["neuroprep-bids-stage"],
            "optional_executor_ids": [],
        }
    )


def test_nfs_worker_stages_bids_and_strips_phi(nfs_tmp: Path) -> None:
    plan = _bids_only_plan(nfs_tmp)
    plan_path = nfs_tmp / "execution-plan.json"
    save_execution_plan(plan, str(plan_path))
    state = RunState(plan.layout.state_path)
    state.initialize(plan, str(plan_path))

    result = run_worker(
        plan.layout.state_path,
        "test-worker",
        {"cpu"},
        max_tasks=1,
    )
    assert result == {"claimed": 1, "completed": 1, "failed": 0}
    summary = state.summary()
    assert summary["counts"] == {"complete": 1}

    task = plan.tasks[0]
    target = Path(task.metadata["bids_path"])
    sidecar = target.with_name(target.name.removesuffix(".nii.gz") + ".json")
    assert target.read_bytes() == b"synthetic-nifti"
    metadata = json.loads(sidecar.read_text())
    assert "PatientName" not in metadata
    assert "StudyDate" not in metadata
    assert "PatientID" not in metadata["Nested"]
    assert metadata["PatientPosition"] == "HFS"
    assert metadata["SourceAcquisitionID"] == task.acquisition_ids[0]
    release = finalize_execution_run(plan, state, str(nfs_tmp / "release"))
    assert release["ready"]
    assert release["task_counts"] == {"complete": 1}
    assert release["artifact_count"] == 1
    assert release["tensor_count"] == 1
    assert Path(release["artifact_index"]).is_file()
    assert Path(release["tensor_index"]).is_file()
    tensor_rows = [
        json.loads(line)
        for line in Path(release["tensor_index"]).read_text().splitlines()
    ]
    assert tensor_rows[0]["tensor_count"] == 1
    assert tensor_rows[1]["path"].endswith(".nii.gz")


def test_failed_task_retries_until_terminal_failure(nfs_tmp: Path) -> None:
    plan = _bids_only_plan(nfs_tmp)
    plan_path = nfs_tmp / "execution-plan.json"
    save_execution_plan(plan, str(plan_path))
    state = RunState(plan.layout.state_path)
    state.initialize(plan, str(plan_path))
    task = state.claim("worker", {"cpu"})
    assert task is not None
    for attempt in range(3):
        state.finish(
            task.task_id,
            "worker",
            return_code=1,
            log_path=str(nfs_tmp / f"failure-{attempt}.log"),
            error="intentional",
        )
        if attempt < 2:
            task = state.claim("worker", {"cpu"})
            assert task is not None
    assert state.summary()["counts"] == {"failed": 1}


def test_container_environment_crosses_cleanenv_with_mounted_paths(
    monkeypatch, nfs_tmp: Path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    work_root = nfs_tmp / "work"
    task = ExecutableTask(
        task_id="container-environment",
        logical_task_ids=["logical"],
        executor_id="smriprep",
        plugin_ids=["mri.t1w"],
        stages=["anat_preprocess"],
        scope=ExecutionScope.SUBJECT,
        subject_id="sub-01",
        command=[
            "/usr/bin/singularity",
            "run",
            "--cleanenv",
            "--bind",
            f"{work_root}:/work",
            "/image.sif",
        ],
        environment={
            "TEMPLATEFLOW_HOME": str(work_root / "cache" / "templateflow"),
            "NXF_HOME": str(work_root / "cache" / "nextflow"),
        },
        output_root=str(nfs_tmp / "derivatives"),
        work_dir=str(work_root / "tasks" / "container-environment"),
        resource_class="gpu",
        cpus=1,
        memory_gb=4,
        gpus=1,
        walltime_minutes=10,
        fingerprint="test",
    )

    environment = _task_environment(task)

    for prefix in ("SINGULARITYENV_", "APPTAINERENV_"):
        assert environment[f"{prefix}TEMPLATEFLOW_HOME"] == "/work/cache/templateflow"
        assert environment[f"{prefix}NXF_HOME"] == "/work/cache/nextflow"
        assert environment[f"{prefix}CUDA_VISIBLE_DEVICES"] == "2"
        assert environment[f"{prefix}NVIDIA_VISIBLE_DEVICES"] == "2"


def test_deepprep_view_contains_only_the_selected_primary_t1(nfs_tmp: Path) -> None:
    bids_root = nfs_tmp / "bids"
    primary = bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_acq-primary_T1w.nii.gz"
    secondary = bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_acq-secondary_T1w.nii.gz"
    primary.parent.mkdir(parents=True)
    primary.write_bytes(b"primary")
    primary.with_name(primary.name.removesuffix(".nii.gz") + ".json").write_text("{}\n")
    secondary.write_bytes(b"secondary")
    (bids_root / "dataset_description.json").write_text(
        '{"Name":"test","BIDSVersion":"1.10.0"}\n'
    )
    task = ExecutableTask(
        task_id="deepprep-view",
        logical_task_ids=["logical"],
        executor_id="deepprep",
        plugin_ids=["mri.t1w"],
        stages=["anat_segment"],
        scope=ExecutionScope.SUBJECT,
        subject_id="sub-01",
        command=["true"],
        output_root=str(nfs_tmp / "derivatives"),
        work_dir=str(nfs_tmp / "work" / "task"),
        resource_class="gpu",
        cpus=1,
        memory_gb=4,
        gpus=1,
        walltime_minutes=10,
        fingerprint="test",
        metadata={
            "bids_root": str(bids_root),
            "primary_anatomical_bids_input": str(primary),
        },
    )

    _prepare_deepprep_bids_view(task)

    view = Path(task.work_dir) / "deepprep-bids"
    assert (view / primary.relative_to(bids_root)).is_symlink()
    assert not (view / secondary.relative_to(bids_root)).exists()
    assert (view / "dataset_description.json").is_file()


def test_smriprep_view_contains_only_the_selected_primary_t1(nfs_tmp: Path) -> None:
    bids_root = nfs_tmp / "bids"
    primary = bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz"
    secondary = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_acq-post_T1w.nii.gz"
    )
    primary.parent.mkdir(parents=True)
    primary.write_bytes(b"primary")
    secondary.write_bytes(b"secondary")
    (bids_root / "dataset_description.json").write_text(
        '{"Name":"test","BIDSVersion":"1.10.0"}\n'
    )
    task = ExecutableTask(
        task_id="smriprep-view",
        logical_task_ids=["logical"],
        executor_id="smriprep",
        plugin_ids=["mri.t1w"],
        stages=["anat_preprocess"],
        scope=ExecutionScope.SUBJECT,
        subject_id="sub-01",
        command=["true"],
        output_root=str(nfs_tmp / "derivatives"),
        work_dir=str(nfs_tmp / "work" / "smriprep-task"),
        resource_class="cpu",
        cpus=1,
        memory_gb=4,
        gpus=0,
        walltime_minutes=10,
        fingerprint="test",
        metadata={
            "bids_root": str(bids_root),
            "primary_anatomical_bids_input": str(primary),
        },
    )

    _prepare_primary_anatomical_bids_view(task)

    view = Path(task.work_dir) / "smriprep-bids"
    assert (view / primary.relative_to(bids_root)).is_symlink()
    assert not (view / secondary.relative_to(bids_root)).exists()
