from pathlib import Path

import pytest

from neuroprep_agentic.execution_models import (
    ExecutableTask,
    ExecutionPlan,
    ExecutionScope,
    RunLayout,
)
from neuroprep_agentic.qualification import compile_backend_evidence
from neuroprep_agentic.state import RunState


def _qualification(nfs_tmp: Path) -> tuple[ExecutionPlan, RunState, Path]:
    artifact = nfs_tmp / "derivatives" / "result.nii.gz"
    layout = RunLayout(
        dataset_id="qualification-test",
        source_root=str(nfs_tmp / "source"),
        bids_root=str(nfs_tmp / "bids"),
        derivatives_root=str(nfs_tmp / "derivatives"),
        work_root=str(nfs_tmp / "work"),
        state_path=str(nfs_tmp / "state" / "tasks.sqlite"),
    )
    task = ExecutableTask(
        task_id="qualified-task",
        logical_task_ids=["logical"],
        executor_id="qsiprep",
        plugin_ids=["mri.dwi"],
        stages=["dwi_preprocess"],
        scope=ExecutionScope.SUBJECT,
        subject_id="sub-01",
        command=["true"],
        output_root=layout.derivatives_root,
        work_dir=str(nfs_tmp / "work" / "task"),
        resource_class="gpu",
        cpus=1,
        memory_gb=4,
        gpus=1,
        walltime_minutes=10,
        fingerprint="test",
    )
    plan = ExecutionPlan(
        dataset_id=layout.dataset_id,
        generated_at="2026-01-01T00:00:00Z",
        catalog_digest="catalog",
        layout=layout,
        tasks=[task],
        required_executor_ids=["qsiprep"],
        optional_executor_ids=[],
        planned_smoke_tests=[
            {
                "test_id": "smoke--dwi_preprocess--qsiprep",
                "executor_id": "qsiprep",
            }
        ],
    )
    plan_path = nfs_tmp / "execution-plan.json"
    plan_path.write_text(plan.model_dump_json())
    state = RunState(layout.state_path)
    state.initialize(plan, str(plan_path))
    return plan, state, artifact


def test_qualification_evidence_requires_terminal_tasks(nfs_tmp: Path) -> None:
    plan, state, _artifact = _qualification(nfs_tmp)
    with pytest.raises(ValueError, match="before qsiprep is terminal"):
        compile_backend_evidence(plan, state)


def test_qualification_evidence_passes_only_with_persistent_artifacts(nfs_tmp: Path) -> None:
    plan, state, artifact = _qualification(nfs_tmp)
    claimed = state.claim("worker", {"gpu"})
    assert claimed is not None
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"result")
    log = nfs_tmp / "work" / "task" / "attempt.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("ok\n")
    state.finish(
        claimed.task_id,
        "worker",
        return_code=0,
        log_path=str(log),
        artifacts=[str(artifact)],
    )
    assert compile_backend_evidence(plan, state) == {
        "smoke--dwi_preprocess--qsiprep": "passed"
    }
    artifact.unlink()
    assert compile_backend_evidence(plan, state) == {
        "smoke--dwi_preprocess--qsiprep": "failed"
    }
