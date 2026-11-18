from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from neuroprep_agentic import main as main_module
from neuroprep_agentic.execution_models import (
    ArtifactRequirement,
    ExecutableTask,
    ExecutionPlan,
    ExecutionScope,
    RunLayout,
)
from neuroprep_agentic.state import RunState


def _task(
    root: Path,
    task_id: str,
    resource_class: str,
    *,
    depends_on: list[str] | None = None,
) -> ExecutableTask:
    return ExecutableTask(
        task_id=task_id,
        logical_task_ids=[f"logical-{task_id}"],
        executor_id=f"executor-{task_id}",
        plugin_ids=["test.plugin"],
        stages=["test"],
        scope=ExecutionScope.SUBJECT,
        subject_id=f"sub-{task_id}",
        depends_on=depends_on or [],
        command=["true"],
        output_root=str(root / "derivatives"),
        work_dir=str(root / "work" / task_id),
        resource_class=resource_class,
        cpus=1,
        memory_gb=1,
        gpus=1 if resource_class == "gpu" else 0,
        walltime_minutes=10,
        artifacts=[
            ArtifactRequirement(
                kind="test-artifact",
                patterns=(f"{task_id}/artifact*",),
            )
        ],
        fingerprint=f"fingerprint-{task_id}",
    )


def _initialized_state(root: Path, tasks: list[ExecutableTask]) -> RunState:
    layout = RunLayout(
        dataset_id="recovery-test",
        source_root=str(root / "source"),
        bids_root=str(root / "bids"),
        derivatives_root=str(root / "derivatives"),
        work_root=str(root / "work"),
        state_path=str(root / "state" / "tasks.sqlite"),
    )
    plan = ExecutionPlan(
        dataset_id="recovery-test",
        generated_at=datetime.now(UTC).isoformat(),
        catalog_digest="recovery-test-catalog",
        layout=layout,
        tasks=tasks,
        required_executor_ids=sorted({task.executor_id for task in tasks}),
        optional_executor_ids=[],
    )
    plan_path = root / "execution-plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    state = RunState(layout.state_path)
    state.initialize(plan, str(plan_path))
    return state


def _expire_claim(state: RunState, task_id: str) -> tuple[str, str, str]:
    started_at = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    lease_expires_at = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    finished_at = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
    with sqlite3.connect(state.path) as connection:
        attempt = connection.execute(
            "SELECT attempts FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()[0]
        connection.execute(
            "UPDATE tasks SET lease_expires_at=?,updated_at=? WHERE task_id=?",
            (lease_expires_at, started_at, task_id),
        )
        connection.execute(
            "UPDATE attempts SET started_at=? WHERE task_id=? AND attempt=?",
            (started_at, task_id, attempt),
        )
        connection.commit()
    return started_at, lease_expires_at, finished_at


def _row(state: RunState, table: str, task_id: str) -> sqlite3.Row:
    with sqlite3.connect(state.path) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            f"SELECT * FROM {table} WHERE task_id=? ORDER BY rowid DESC LIMIT 1",
            (task_id,),
        ).fetchone()


def test_claim_recovers_only_requested_resource_classes(nfs_tmp: Path) -> None:
    state = _initialized_state(
        nfs_tmp,
        [_task(nfs_tmp, "cpu-task", "cpu"), _task(nfs_tmp, "gpu-task", "gpu")],
    )
    assert state.claim("old-cpu-worker", {"cpu"}) is not None
    assert state.claim("old-gpu-worker", {"gpu"}) is not None
    _expire_claim(state, "cpu-task")
    _expire_claim(state, "gpu-task")
    gpu_task_before = dict(_row(state, "tasks", "gpu-task"))
    gpu_attempt_before = dict(_row(state, "attempts", "gpu-task"))

    assert state.claim("new-cpu-worker", {"cpu"}) is not None

    cpu = _row(state, "tasks", "cpu-task")
    gpu = _row(state, "tasks", "gpu-task")
    assert (cpu["status"], cpu["claimed_by"], cpu["attempts"]) == (
        "running",
        "new-cpu-worker",
        2,
    )
    assert (gpu["status"], gpu["claimed_by"], gpu["attempts"]) == (
        "running",
        "old-gpu-worker",
        1,
    )
    assert dict(gpu) == gpu_task_before
    assert dict(_row(state, "attempts", "gpu-task")) == gpu_attempt_before
    with sqlite3.connect(state.path) as connection:
        cpu_attempts = connection.execute(
            "SELECT attempt,status FROM attempts WHERE task_id='cpu-task' ORDER BY attempt"
        ).fetchall()
        gpu_attempts = connection.execute(
            "SELECT attempt,status FROM attempts WHERE task_id='gpu-task' ORDER BY attempt"
        ).fetchall()
    assert cpu_attempts == [(1, "failed"), (2, "running")]
    assert gpu_attempts == [(1, "running")]


def test_admin_apply_does_not_create_a_mistyped_state_database(nfs_tmp: Path) -> None:
    missing_state = nfs_tmp / "state" / "mistyped.sqlite"
    state = RunState(str(missing_state))

    with pytest.raises(FileNotFoundError):
        state.reconcile_blocked_children({"child": "parent"}, apply=True)

    assert not missing_state.exists()


def test_recover_stale_task_cli_is_explicit_and_rejects_repeated_apply(
    nfs_tmp: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state = _initialized_state(nfs_tmp, [_task(nfs_tmp, "gpu-task", "gpu")])
    assert state.claim("slurm-123-gpu", {"gpu"}) is not None
    _, lease_expires_at, finished_at = _expire_claim(state, "gpu-task")
    task_log = nfs_tmp / "work" / "gpu-task" / "attempt.log"
    task_log.parent.mkdir(parents=True, exist_ok=True)
    task_log.write_text("scheduler terminated the allocation\n", encoding="utf-8")
    arguments = [
        "recover-stale-task",
        "--state",
        str(state.path),
        "--task-id",
        "gpu-task",
        "--expected-worker-id",
        "slurm-123-gpu",
        "--expected-attempt",
        "1",
        "--expected-lease-expires-at",
        lease_expires_at,
        "--finished-at",
        finished_at,
        "--reason",
        "Slurm job 123 reached its time limit",
        "--log-path",
        str(task_log),
    ]

    with pytest.raises(SystemExit) as missing_mode:
        main_module.cli(arguments)
    assert missing_mode.value.code == 2
    capsys.readouterr()
    before_rejected_apply = {
        "task": dict(_row(state, "tasks", "gpu-task")),
        "attempt": dict(_row(state, "attempts", "gpu-task")),
    }
    with pytest.raises(RuntimeError, match="task worker differs"):
        state.recover_stale_task(
            "gpu-task",
            expected_worker_id="wrong-worker",
            expected_attempt=1,
            expected_lease_expires_at=lease_expires_at,
            finished_at=finished_at,
            reason="must not apply",
            log_path=str(task_log),
            apply=True,
        )
    assert dict(_row(state, "tasks", "gpu-task")) == before_rejected_apply["task"]
    assert dict(_row(state, "attempts", "gpu-task")) == before_rejected_apply["attempt"]

    main_module.cli([*arguments, "--dry-run"])
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["mode"] == "dry-run"
    assert dry_run["applied"] is False
    assert _row(state, "tasks", "gpu-task")["status"] == "running"
    assert _row(state, "attempts", "gpu-task")["status"] == "running"

    main_module.cli([*arguments, "--apply"])
    applied = json.loads(capsys.readouterr().out)
    assert applied["mode"] == "apply"
    assert applied["applied"] is True
    task = _row(state, "tasks", "gpu-task")
    attempt = _row(state, "attempts", "gpu-task")
    assert (task["status"], task["claimed_by"], task["lease_expires_at"]) == (
        "pending",
        None,
        None,
    )
    assert task["attempts"] == 1
    assert task["last_error"] == "Slurm job 123 reached its time limit"
    assert task["log_path"] == str(task_log)
    assert attempt["status"] == "failed"
    assert attempt["finished_at"] == finished_at
    assert attempt["log_path"] == str(task_log)

    with pytest.raises(RuntimeError, match="task is not running"):
        state.recover_stale_task(
            "gpu-task",
            expected_worker_id="slurm-123-gpu",
            expected_attempt=1,
            expected_lease_expires_at=lease_expires_at,
            finished_at=finished_at,
            reason="Slurm job 123 reached its time limit",
            log_path=str(task_log),
            apply=True,
        )


def test_recover_stale_retry_replaces_prior_task_return_code(nfs_tmp: Path) -> None:
    state = _initialized_state(nfs_tmp, [_task(nfs_tmp, "retry-task", "gpu")])
    assert state.claim("first-worker", {"gpu"}) is not None
    first_log = nfs_tmp / "work" / "retry-task" / "first.log"
    first_log.parent.mkdir(parents=True, exist_ok=True)
    first_log.write_text("first attempt failed\n", encoding="utf-8")
    state.finish(
        "retry-task",
        "first-worker",
        return_code=17,
        log_path=str(first_log),
        error="first attempt failed",
    )
    assert state.claim("second-worker", {"gpu"}) is not None
    _, lease_expires_at, finished_at = _expire_claim(state, "retry-task")
    assert _row(state, "tasks", "retry-task")["return_code"] == 17
    recovery_log = nfs_tmp / "work" / "retry-task" / "recovery.log"
    recovery_log.write_text("second worker lease expired\n", encoding="utf-8")

    result = state.recover_stale_task(
        "retry-task",
        expected_worker_id="second-worker",
        expected_attempt=2,
        expected_lease_expires_at=lease_expires_at,
        finished_at=finished_at,
        reason="second worker lease expired",
        log_path=str(recovery_log),
        apply=True,
    )

    assert result["task_before"]["return_code"] == 17
    assert _row(state, "tasks", "retry-task")["return_code"] is None
    assert _row(state, "attempts", "retry-task")["status"] == "failed"


def test_reconcile_blocked_child_requires_valid_parent_artifacts_and_cas(
    nfs_tmp: Path,
) -> None:
    parent = _task(nfs_tmp, "parent", "gpu")
    secondary = _task(nfs_tmp, "secondary", "low_memory")
    child = _task(
        nfs_tmp,
        "child",
        "high_memory",
        depends_on=[parent.task_id, secondary.task_id],
    )
    protected = _task(nfs_tmp, "protected", "cpu")
    state = _initialized_state(nfs_tmp, [parent, secondary, child, protected])
    assert state.claim("parent-worker", {"gpu"}) is not None
    parent_log = nfs_tmp / "work" / "parent" / "attempt.log"
    parent_log.parent.mkdir(parents=True, exist_ok=True)
    parent_log.write_text("complete\n", encoding="utf-8")
    artifact = nfs_tmp / "derivatives" / "parent" / "artifact.nii.gz"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"valid-artifact")
    state.finish(
        parent.task_id,
        "parent-worker",
        return_code=0,
        log_path=str(parent_log),
        artifacts=[str(artifact)],
    )
    assert state.claim("secondary-worker", {"low_memory"}) is not None
    secondary_log = nfs_tmp / "work" / "secondary" / "attempt.log"
    secondary_log.parent.mkdir(parents=True, exist_ok=True)
    secondary_log.write_text("complete\n", encoding="utf-8")
    secondary_artifact = nfs_tmp / "derivatives" / "secondary" / "artifact-directory"
    secondary_artifact.mkdir(parents=True, exist_ok=True)
    (secondary_artifact / "empty-directory").mkdir()
    secondary_result = secondary_artifact / "result.nii.gz"
    secondary_result.write_bytes(b"")
    state.finish(
        secondary.task_id,
        "secondary-worker",
        return_code=0,
        log_path=str(secondary_log),
        artifacts=[str(secondary_artifact)],
    )
    with sqlite3.connect(state.path) as connection:
        connection.execute(
            "UPDATE tasks SET status='blocked',last_error='dependency failed' WHERE task_id=?",
            (child.task_id,),
        )
        connection.commit()
    protected_before = dict(_row(state, "tasks", protected.task_id))

    with pytest.raises(RuntimeError, match="artifact is empty"):
        state.reconcile_blocked_children({child.task_id: parent.task_id}, apply=False)
    assert _row(state, "tasks", child.task_id)["status"] == "blocked"

    secondary_result.write_bytes(b"valid-secondary-artifact")
    with sqlite3.connect(state.path) as connection:
        connection.execute(
            "UPDATE attempts SET return_code=7 WHERE task_id=? AND attempt=1",
            (secondary.task_id,),
        )
        connection.commit()
    with pytest.raises(RuntimeError, match="matching successful attempt"):
        state.reconcile_blocked_children({child.task_id: parent.task_id}, apply=False)
    with sqlite3.connect(state.path) as connection:
        connection.execute(
            "UPDATE attempts SET return_code=0 WHERE task_id=? AND attempt=1",
            (secondary.task_id,),
        )
        connection.commit()
    dry_run = state.reconcile_blocked_children(
        {child.task_id: parent.task_id},
        apply=False,
    )
    assert dry_run["mode"] == "dry-run"
    assert dry_run["candidate_count"] == 1
    assert _row(state, "tasks", child.task_id)["status"] == "blocked"

    applied = state.reconcile_blocked_children(
        {child.task_id: parent.task_id},
        apply=True,
    )
    assert applied["applied"] is True
    reconciled = _row(state, "tasks", child.task_id)
    assert reconciled["status"] == "pending"
    assert reconciled["last_error"] is None
    assert dict(_row(state, "tasks", protected.task_id)) == protected_before

    with pytest.raises(RuntimeError, match="unattempted dependency-failed"):
        state.reconcile_blocked_children({child.task_id: parent.task_id}, apply=True)
