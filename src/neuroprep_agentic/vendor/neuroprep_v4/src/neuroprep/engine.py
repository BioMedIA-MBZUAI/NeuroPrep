from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import socket
import sqlite3
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from .execution import topological_tasks
from .provenance import canonical_json, task_fingerprint


TERMINAL_STATUSES = {"complete", "failed", "blocked"}
TRANSIENT_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"preempt",
        r"node fail",
        r"connection reset",
        r"stale file handle",
        r"temporarily unavailable",
        r"transport endpoint is not connected",
        r"no space left on device",
        r"disk quota exceeded",
        r"input/output error",
    )
)


@dataclass(frozen=True)
class ClaimedTask:
    spec: dict[str, Any]
    attempt: int


@dataclass(frozen=True)
class WorkerResult:
    completed: int
    failed: int
    blocked: int
    worker_id: str


class TaskDatabase:
    """A short-transaction task store guarded by a POSIX file lock.

    The external lock avoids SQLite WAL assumptions on shared filesystems. Task
    commands execute outside the lock, so the serialized section is limited to
    claims, heartbeats, and state transitions.
    """

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @contextmanager
    def _locked_connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            connection = sqlite3.connect(self.path, timeout=60)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("PRAGMA synchronous = FULL")
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()
        finally:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)

    def initialize(self, plan: dict[str, Any]) -> int:
        ordered = topological_tasks(plan)
        with self._locked_connection() as connection:
            self._create_schema(connection)
            for task in ordered:
                fingerprint = task.get("fingerprint") or task_fingerprint(task)
                existing = connection.execute(
                    "SELECT fingerprint FROM tasks WHERE task_id = ?", (task["task_id"],)
                ).fetchone()
                if existing and existing["fingerprint"] != fingerprint:
                    raise ValueError(
                        f"Task {task['task_id']} changed scientific fingerprint; use a new state "
                        "database or a new task identity"
                    )
                if existing:
                    continue
                resources = task.get("resources", {})
                resource_class = "gpu" if int(resources.get("gpus", 0)) > 0 else "cpu"
                command = task.get("command") or []
                status = "pending" if command else "blocked"
                error = None if command else "No reviewed command is configured"
                max_attempts = max(1, int(task.get("params", {}).get("max_attempts", 2)))
                now = _now()
                connection.execute(
                    """
                    INSERT INTO tasks (
                        task_id, dataset_id, subject, session, stage, pipeline, resource_class,
                        fingerprint, spec_json, status, attempts, max_attempts, created_at,
                        updated_at, last_error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                    """,
                    (
                        task["task_id"],
                        task.get("dataset_id"),
                        task.get("subject"),
                        task.get("session"),
                        task.get("stage"),
                        task.get("pipeline"),
                        resource_class,
                        fingerprint,
                        canonical_json(task),
                        status,
                        max_attempts,
                        now,
                        now,
                        error,
                    ),
                )
                for dependency in task.get("dependencies", []):
                    connection.execute(
                        "INSERT INTO dependencies (task_id, dependency_id) VALUES (?, ?)",
                        (task["task_id"], dependency),
                    )
            self._propagate_blocked(connection)
        return len(ordered)

    def claim(
        self,
        worker_id: str,
        resource_class: str,
        lease_seconds: int = 300,
    ) -> ClaimedTask | None:
        if resource_class not in {"cpu", "gpu", "any"}:
            raise ValueError("resource_class must be cpu, gpu, or any")
        with self._locked_connection() as connection:
            self._create_schema(connection)
            self._reap_expired(connection)
            self._propagate_blocked(connection)
            resource_clause = "1 = 1" if resource_class == "any" else "t.resource_class = ?"
            parameters: tuple[Any, ...] = () if resource_class == "any" else (resource_class,)
            row = connection.execute(
                f"""
                SELECT t.* FROM tasks t
                WHERE t.status = 'pending'
                  AND {resource_clause}
                  AND NOT EXISTS (
                      SELECT 1 FROM dependencies d
                      JOIN tasks parent ON parent.task_id = d.dependency_id
                      WHERE d.task_id = t.task_id AND parent.status != 'complete'
                  )
                ORDER BY
                    CASE t.stage
                        WHEN 'ingest' THEN 0 WHEN 'validate' THEN 1
                        WHEN 'preprocess' THEN 2 WHEN 'denoise' THEN 3
                        WHEN 'postprocess' THEN 4 WHEN 'register' THEN 5
                        WHEN 'qc' THEN 6 WHEN 'release' THEN 7
                        WHEN 'index' THEN 8 WHEN 'grid' THEN 9
                        WHEN 'assemble' THEN 10 WHEN 'align' THEN 11
                        WHEN 'tensor' THEN 12 WHEN 'link' THEN 13
                        WHEN 'split' THEN 14 WHEN 'harmonize' THEN 15
                        WHEN 'audit' THEN 16 ELSE 17
                    END,
                    t.task_id
                LIMIT 1
                """,
                parameters,
            ).fetchone()
            if row is None:
                return None
            attempt = int(row["attempts"]) + 1
            lease_expires = (
                datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
            ).isoformat()
            updated = connection.execute(
                """
                UPDATE tasks
                SET status = 'running', attempts = ?, claimed_by = ?, lease_expires_at = ?,
                    updated_at = ?, last_error = NULL
                WHERE task_id = ? AND status = 'pending'
                """,
                (attempt, worker_id, lease_expires, _now(), row["task_id"]),
            )
            if updated.rowcount != 1:
                return None
            connection.execute(
                """
                INSERT INTO attempts (task_id, attempt, worker_id, started_at, status)
                VALUES (?, ?, ?, ?, 'running')
                """,
                (row["task_id"], attempt, worker_id, _now()),
            )
            return ClaimedTask(json.loads(row["spec_json"]), attempt)

    def heartbeat(self, task_id: str, worker_id: str, lease_seconds: int) -> bool:
        lease_expires = (
            datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
        ).isoformat()
        with self._locked_connection() as connection:
            updated = connection.execute(
                """
                UPDATE tasks SET lease_expires_at = ?, updated_at = ?
                WHERE task_id = ? AND status = 'running' AND claimed_by = ?
                """,
                (lease_expires, _now(), task_id, worker_id),
            )
            return updated.rowcount == 1

    def finish(
        self,
        task_id: str,
        worker_id: str,
        attempt: int,
        return_code: int,
        log_path: Path,
        error: str | None,
        transient: bool,
    ) -> str:
        with self._locked_connection() as connection:
            row = connection.execute(
                """
                SELECT attempts, max_attempts, pipeline, stage, spec_json
                FROM tasks WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown task: {task_id}")
            if return_code == 0 and error is None:
                status = "complete"
            elif transient and int(row["attempts"]) < int(row["max_attempts"]):
                status = "pending"
            else:
                status = "failed"
            attempt_status = "failed-transient" if status == "pending" else status
            failure_signature = (
                _failure_signature(error or "") if status == "failed" else None
            )
            connection.execute(
                """
                UPDATE tasks
                SET status = ?, claimed_by = NULL, lease_expires_at = NULL, updated_at = ?,
                    return_code = ?, last_error = ?, log_path = ?, failure_signature = ?
                WHERE task_id = ? AND claimed_by = ?
                """,
                (
                    status,
                    _now(),
                    return_code,
                    error,
                    str(log_path),
                    failure_signature,
                    task_id,
                    worker_id,
                ),
            )
            connection.execute(
                """
                UPDATE attempts
                SET finished_at = ?, status = ?, return_code = ?, error = ?, log_path = ?
                WHERE task_id = ? AND attempt = ?
                """,
                (_now(), attempt_status, return_code, error, str(log_path), task_id, attempt),
            )
            if status == "failed" and failure_signature:
                self._apply_systemic_failure_circuit(
                    connection,
                    pipeline=str(row["pipeline"]),
                    stage=str(row["stage"]),
                    signature=failure_signature,
                    spec=json.loads(row["spec_json"]),
                )
            self._propagate_blocked(connection)
            return status

    def summary(self) -> dict[str, Any]:
        with self._locked_connection() as connection:
            self._create_schema(connection)
            self._reap_expired(connection)
            self._propagate_blocked(connection)
            status_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status ORDER BY status"
            ).fetchall()
            resource_rows = connection.execute(
                """
                SELECT resource_class, status, COUNT(*) AS count
                FROM tasks GROUP BY resource_class, status ORDER BY resource_class, status
                """
            ).fetchall()
            failures = connection.execute(
                """
                SELECT task_id, status, attempts, last_error, log_path, failure_signature
                FROM tasks WHERE status IN ('failed', 'blocked') ORDER BY task_id LIMIT 50
                """
            ).fetchall()
            total = connection.execute("SELECT COUNT(*) AS count FROM tasks").fetchone()["count"]
            return {
                "database": str(self.path),
                "total": total,
                "status_counts": {row["status"]: row["count"] for row in status_rows},
                "resource_status_counts": {
                    f"{row['resource_class']}:{row['status']}": row["count"]
                    for row in resource_rows
                },
                "terminal": sum(
                    row["count"] for row in status_rows if row["status"] in TERMINAL_STATUSES
                ),
                "issues": [dict(row) for row in failures],
            }

    def has_unfinished(self, resource_class: str) -> bool:
        with self._locked_connection() as connection:
            self._reap_expired(connection)
            self._propagate_blocked(connection)
            if resource_class == "any":
                row = connection.execute(
                    "SELECT 1 FROM tasks WHERE status IN ('pending', 'running') LIMIT 1"
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT 1 FROM tasks
                    WHERE resource_class = ? AND status IN ('pending', 'running') LIMIT 1
                    """,
                    (resource_class,),
                ).fetchone()
            return row is not None

    def task_rows(self) -> list[dict[str, Any]]:
        with self._locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT task_id, dataset_id, subject, session, stage, pipeline, resource_class,
                       fingerprint, status, attempts, max_attempts, return_code, created_at,
                       updated_at, last_error, log_path, failure_signature, spec_json
                FROM tasks ORDER BY task_id
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def retry_failed(self, task_ids: list[str]) -> dict[str, Any]:
        """Retry explicit failed roots and reopen their circuit-blocked cohort.

        An explicit retry is an operator acknowledgement that the root cause has
        been repaired. Tasks blocked by the same systemic-failure signature are
        therefore safe to reopen along with ordinary propagated descendants.
        """

        if not task_ids:
            raise ValueError("At least one failed task id is required")
        requested = sorted(set(task_ids))
        with self._locked_connection() as connection:
            self._create_schema(connection)
            rows = connection.execute(
                f"""
                SELECT task_id, status, attempts, max_attempts, pipeline, stage,
                       failure_signature
                FROM tasks WHERE task_id IN ({",".join("?" for _ in requested)})
                """,
                requested,
            ).fetchall()
            observed = {str(row["task_id"]): row for row in rows}
            missing = [task_id for task_id in requested if task_id not in observed]
            if missing:
                raise ValueError(f"Unknown task ids: {', '.join(missing)}")
            invalid = [
                task_id
                for task_id in requested
                if str(observed[task_id]["status"]) != "failed"
            ]
            if invalid:
                raise ValueError(
                    "Only failed task roots can be retried: " + ", ".join(invalid)
                )
            circuit_keys = {
                (
                    str(observed[task_id]["pipeline"]),
                    str(observed[task_id]["stage"]),
                    str(observed[task_id]["failure_signature"]),
                )
                for task_id in requested
                if observed[task_id]["failure_signature"]
            }
            for task_id in requested:
                connection.execute(
                    """
                    UPDATE tasks
                    SET status = 'pending', claimed_by = NULL, lease_expires_at = NULL,
                        updated_at = ?, return_code = NULL, last_error = NULL,
                        log_path = NULL, failure_signature = NULL,
                        max_attempts = MAX(max_attempts, attempts + 1)
                    WHERE task_id = ? AND status = 'failed'
                    """,
                    (_now(), task_id),
                )
            reopened_circuit = 0
            for pipeline, stage, signature in sorted(circuit_keys):
                reopened_circuit += connection.execute(
                    """
                    UPDATE tasks
                    SET status = 'pending', claimed_by = NULL, lease_expires_at = NULL,
                        updated_at = ?, return_code = NULL, last_error = NULL,
                        log_path = NULL, failure_signature = NULL
                    WHERE pipeline = ? AND stage = ? AND status = 'blocked'
                      AND failure_signature = ?
                      AND last_error LIKE 'Systemic failure circuit opened%'
                    """,
                    (_now(), pipeline, stage, signature),
                ).rowcount
            reopened = 0
            while True:
                changed = connection.execute(
                    """
                    UPDATE tasks
                    SET status = 'pending', updated_at = ?, last_error = NULL,
                        failure_signature = NULL
                    WHERE status = 'blocked'
                      AND last_error = 'A required upstream task failed or is blocked'
                      AND NOT EXISTS (
                          SELECT 1 FROM dependencies d
                          JOIN tasks parent ON parent.task_id = d.dependency_id
                          WHERE d.task_id = tasks.task_id
                            AND parent.status IN ('failed', 'blocked')
                      )
                    """,
                    (_now(),),
                ).rowcount
                reopened += changed
                if not changed:
                    break
            return {
                "status": "pass",
                "retried": requested,
                "reopened_circuit_tasks": reopened_circuit,
                "reopened_descendants": reopened,
            }

    def recover_workers(
        self,
        worker_prefixes: list[str],
        reason: str = "Worker pool was intentionally stopped",
    ) -> dict[str, Any]:
        """Requeue claims from confirmed-stopped worker pools.

        Callers must verify that the corresponding scheduler jobs have stopped
        before invoking this method. Prefix matching supports pools whose worker
        ids append a resource class and array index to the scheduler job id.
        """

        prefixes = sorted({value.strip() for value in worker_prefixes if value.strip()})
        if not prefixes:
            raise ValueError("At least one worker id prefix is required")
        with self._locked_connection() as connection:
            self._create_schema(connection)
            clauses = " OR ".join("claimed_by LIKE ?" for _ in prefixes)
            parameters = [f"{prefix}%" for prefix in prefixes]
            rows = connection.execute(
                f"""
                SELECT task_id, attempts, claimed_by FROM tasks
                WHERE status = 'running' AND ({clauses})
                """,
                parameters,
            ).fetchall()
            now = _now()
            for row in rows:
                connection.execute(
                    """
                    UPDATE tasks
                    SET status = 'pending', claimed_by = NULL, lease_expires_at = NULL,
                        updated_at = ?, return_code = NULL, last_error = ?,
                        failure_signature = NULL,
                        max_attempts = MAX(max_attempts, attempts + 1)
                    WHERE task_id = ? AND status = 'running'
                    """,
                    (now, reason, row["task_id"]),
                )
                connection.execute(
                    """
                    UPDATE attempts
                    SET finished_at = ?, status = 'aborted', error = ?
                    WHERE task_id = ? AND attempt = ? AND status = 'running'
                    """,
                    (now, reason, row["task_id"], row["attempts"]),
                )
            return {
                "status": "pass",
                "worker_prefixes": prefixes,
                "requeued": len(rows),
                "task_ids": [str(row["task_id"]) for row in rows],
            }

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                dataset_id TEXT,
                subject TEXT,
                session TEXT,
                stage TEXT NOT NULL,
                pipeline TEXT NOT NULL,
                resource_class TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                spec_json TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 2,
                claimed_by TEXT,
                lease_expires_at TEXT,
                return_code INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_error TEXT,
                log_path TEXT,
                failure_signature TEXT
            );
            CREATE TABLE IF NOT EXISTS dependencies (
                task_id TEXT NOT NULL,
                dependency_id TEXT NOT NULL,
                PRIMARY KEY (task_id, dependency_id),
                FOREIGN KEY (task_id) REFERENCES tasks(task_id),
                FOREIGN KEY (dependency_id) REFERENCES tasks(task_id)
            );
            CREATE TABLE IF NOT EXISTS attempts (
                task_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                worker_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                return_code INTEGER,
                error TEXT,
                log_path TEXT,
                PRIMARY KEY (task_id, attempt),
                FOREIGN KEY (task_id) REFERENCES tasks(task_id)
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_ready
                ON tasks(status, resource_class, stage, task_id);
            CREATE INDEX IF NOT EXISTS idx_dependencies_task ON dependencies(task_id);
            """
        )
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
        }
        if "failure_signature" not in columns:
            connection.execute(
                "ALTER TABLE tasks ADD COLUMN failure_signature TEXT"
            )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_failure
            ON tasks(pipeline, stage, status, failure_signature)
            """
        )

    @staticmethod
    def _apply_systemic_failure_circuit(
        connection: sqlite3.Connection,
        pipeline: str,
        stage: str,
        signature: str,
        spec: dict[str, Any],
    ) -> None:
        params = spec.get("params", {})
        if bool(params.get("disable_systemic_failure_circuit", False)):
            return
        if stage not in {"preprocess", "denoise"}:
            return
        threshold = max(2, int(params.get("systemic_failure_threshold", 3)))
        failures = connection.execute(
            """
            SELECT COUNT(*) AS count FROM tasks
            WHERE pipeline = ? AND stage = ? AND status = 'failed'
              AND failure_signature = ?
            """,
            (pipeline, stage, signature),
        ).fetchone()["count"]
        if int(failures) < threshold:
            return
        reason = (
            f"Systemic failure circuit opened for {pipeline} after {failures} "
            f"matching failures (signature={signature})"
        )
        connection.execute(
            """
            UPDATE tasks
            SET status = 'blocked', updated_at = ?, last_error = ?,
                failure_signature = ?
            WHERE pipeline = ? AND stage = ? AND status = 'pending'
            """,
            (_now(), reason, signature, pipeline, stage),
        )

    @staticmethod
    def _propagate_blocked(connection: sqlite3.Connection) -> None:
        while True:
            changed = connection.execute(
                """
                UPDATE tasks
                SET status = 'blocked', updated_at = ?,
                    last_error = 'A required upstream task failed or is blocked'
                WHERE status = 'pending' AND EXISTS (
                    SELECT 1 FROM dependencies d
                    JOIN tasks parent ON parent.task_id = d.dependency_id
                    WHERE d.task_id = tasks.task_id AND parent.status IN ('failed', 'blocked')
                )
                """,
                (_now(),),
            ).rowcount
            if not changed:
                break

    @staticmethod
    def _reap_expired(connection: sqlite3.Connection) -> None:
        now = _now()
        rows = connection.execute(
            """
            SELECT task_id, attempts, max_attempts FROM tasks
            WHERE status = 'running' AND lease_expires_at < ?
            """,
            (now,),
        ).fetchall()
        for row in rows:
            status = "pending" if row["attempts"] < row["max_attempts"] else "failed"
            connection.execute(
                """
                UPDATE tasks SET status = ?, claimed_by = NULL, lease_expires_at = NULL,
                    updated_at = ?, last_error = 'Worker lease expired'
                WHERE task_id = ?
                """,
                (status, now, row["task_id"]),
            )


def run_worker(
    database: Path,
    work_root: Path,
    resource_class: str,
    worker_id: str | None = None,
    max_tasks: int = 0,
    poll_seconds: float = 5.0,
    idle_exit_seconds: float = 0.0,
    lease_seconds: int = 300,
) -> WorkerResult:
    store = TaskDatabase(database)
    worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
    completed = failed = blocked = processed = 0
    idle_since: float | None = None
    while max_tasks <= 0 or processed < max_tasks:
        claim = store.claim(worker_id, resource_class, lease_seconds)
        if claim is None:
            if not store.has_unfinished(resource_class):
                break
            idle_since = idle_since or time.monotonic()
            if idle_exit_seconds > 0 and time.monotonic() - idle_since >= idle_exit_seconds:
                break
            time.sleep(max(0.1, poll_seconds))
            continue
        idle_since = None
        status = _execute_claim(store, claim, work_root, worker_id, lease_seconds)
        processed += 1
        if status == "complete":
            completed += 1
        elif status == "failed":
            failed += 1
        elif status == "blocked":
            blocked += 1
    return WorkerResult(completed, failed, blocked, worker_id)


def _execute_claim(
    store: TaskDatabase,
    claim: ClaimedTask,
    work_root: Path,
    worker_id: str,
    lease_seconds: int,
) -> str:
    task = claim.spec
    task_id = task["task_id"]
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)
    attempt_dir = work_root / safe_id / f"attempt-{claim.attempt:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    log_path = attempt_dir / "task.log"
    command = [str(value) for value in task.get("command") or []]
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        interval = max(10, min(60, lease_seconds // 3))
        while not stop_heartbeat.wait(interval):
            if not store.heartbeat(task_id, worker_id, lease_seconds):
                return

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    return_code = 1
    error: str | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"task_id={task_id}\n")
            log.write(f"attempt={claim.attempt}\n")
            log.write(f"worker_id={worker_id}\n")
            log.write(f"command={shlex.join(command)}\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=attempt_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=_task_environment(task, attempt_dir),
                check=False,
            )
            return_code = completed.returncode
        if return_code == 0:
            missing = _missing_outputs(task)
            if missing:
                return_code = 66
                error = f"Expected outputs are missing: {', '.join(missing[:10])}"
        if return_code != 0 and error is None:
            error = _log_tail(log_path)
    except Exception as exc:
        return_code = 70
        error = f"Worker execution error: {type(exc).__name__}: {exc}"
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=5)
    transient = _is_transient(return_code, error or "")
    return store.finish(
        task_id,
        worker_id,
        claim.attempt,
        return_code,
        log_path,
        error,
        transient,
    )


def _missing_outputs(task: dict[str, Any]) -> list[str]:
    if not bool(task.get("params", {}).get("verify_outputs", False)):
        return []
    return [str(path) for value in task.get("outputs", []) if not (path := Path(value)).exists()]


def _task_environment(
    task: dict[str, Any],
    attempt_dir: Path | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    cpus = max(1, int(task.get("resources", {}).get("cpus", 1)))
    environment.update(
        {
            "OMP_NUM_THREADS": str(cpus),
            "MKL_NUM_THREADS": str(cpus),
            "OPENBLAS_NUM_THREADS": str(cpus),
            "NUMEXPR_NUM_THREADS": str(cpus),
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(cpus),
            "NEUROPREP_TASK_ID": str(task["task_id"]),
        }
    )
    if attempt_dir is not None:
        runtime_tmp = attempt_dir / "runtime-tmp"
        singularity_tmp = attempt_dir / "singularity-tmp"
        runtime_tmp.mkdir(parents=True, exist_ok=True)
        singularity_tmp.mkdir(parents=True, exist_ok=True)
        bind = f"{runtime_tmp}:/tmp,{runtime_tmp}:/var/tmp"
        environment.update(
            {
                "TMPDIR": str(runtime_tmp),
                "SINGULARITY_TMPDIR": str(singularity_tmp),
                "APPTAINER_TMPDIR": str(singularity_tmp),
                "SINGULARITY_BINDPATH": _append_bind_path(
                    environment.get("SINGULARITY_BINDPATH"), bind
                ),
                "APPTAINER_BINDPATH": _append_bind_path(
                    environment.get("APPTAINER_BINDPATH"), bind
                ),
            }
        )
    return environment


def _append_bind_path(existing: str | None, value: str) -> str:
    return f"{existing},{value}" if existing else value


def _is_transient(return_code: int, error: str) -> bool:
    if return_code in {75, 143}:
        return True
    return any(pattern.search(error) for pattern in TRANSIENT_PATTERNS)


def _failure_signature(error: str) -> str:
    """Return a stable cohort-level signature without retaining identifiers."""

    lowered = error.lower()
    known = (
        ("nextflow_shared_workdir_lock", ("nextflow", "lock", "workdir")),
        ("host_container_glibc_mismatch", ("glibc_", "not found")),
        ("bundled_bids_validator_crash", ("bids-validator", "traceback")),
        ("bundled_bids_validator_crash", ("bids validator", "traceback")),
        ("cuda_runtime_failure", ("cuda", "runtimeerror")),
        ("out_of_memory", ("out of memory",)),
        ("expected_outputs_missing", ("expected outputs are missing",)),
    )
    for label, tokens in known:
        if all(token in lowered for token in tokens):
            return label
    normalized = re.sub(r"/[^\s:]+", "<path>", lowered)
    normalized = re.sub(
        r"\bsub-[a-z0-9_-]+\b|\bses-[a-z0-9_-]+\b",
        "<entity>",
        normalized,
    )
    normalized = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", "<uuid>", normalized
    )
    normalized = re.sub(r"\b\d+\b", "<n>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()[:4000]
    if not normalized:
        return "empty-error"
    return "generic-" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _log_tail(path: Path, max_bytes: int = 16000) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - max_bytes))
            return stream.read().decode("utf-8", errors="replace").strip()
    except OSError as exc:
        return f"Task failed and log could not be read: {exc}"


def write_slurm_pool(
    output_dir: Path,
    database: Path,
    work_root: Path,
    cpu_workers: int,
    gpu_workers: int,
    cpu_partition: str,
    gpu_partition: str,
    cpu_cpus: int = 16,
    cpu_memory_gb: int = 64,
    gpu_cpus: int = 16,
    gpu_memory_gb: int = 64,
) -> list[Path]:
    if cpu_workers < 0 or gpu_workers < 0 or cpu_workers + gpu_workers == 0:
        raise ValueError("At least one CPU or GPU worker is required")
    _validate_pool_capacity(
        database,
        cpu_workers,
        gpu_workers,
        cpu_cpus,
        cpu_memory_gb,
        gpu_cpus,
        gpu_memory_gb,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    written: list[Path] = []
    if cpu_workers:
        path = output_dir / "cpu-pool.sbatch"
        _write_pool_script(
            path,
            "cpu",
            cpu_workers,
            cpu_partition,
            cpu_cpus,
            cpu_memory_gb,
            database,
            work_root,
            log_dir,
            gpu=False,
        )
        written.append(path)
    if gpu_workers:
        path = output_dir / "gpu-pool.sbatch"
        _write_pool_script(
            path,
            "gpu",
            gpu_workers,
            gpu_partition,
            gpu_cpus,
            gpu_memory_gb,
            database,
            work_root,
            log_dir,
            gpu=True,
        )
        written.append(path)
    submit = output_dir / "submit.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.extend(f"sbatch {shlex.quote(str(path))}" for path in written)
    _atomic_write(submit, "\n".join(lines) + "\n", executable=True)
    written.append(submit)
    return written


def _validate_pool_capacity(
    database: Path,
    cpu_workers: int,
    gpu_workers: int,
    cpu_cpus: int,
    cpu_memory_gb: int,
    gpu_cpus: int,
    gpu_memory_gb: int,
) -> None:
    rows = TaskDatabase(database).task_rows()
    for row in rows:
        if row["status"] == "blocked":
            continue
        spec = json.loads(row["spec_json"])
        resources = spec.get("resources", {})
        required_cpus = int(resources.get("cpus", 1))
        required_memory = int(resources.get("memory_gb", 4))
        resource_class = row["resource_class"]
        workers = gpu_workers if resource_class == "gpu" else cpu_workers
        available_cpus = gpu_cpus if resource_class == "gpu" else cpu_cpus
        available_memory = gpu_memory_gb if resource_class == "gpu" else cpu_memory_gb
        if workers == 0:
            raise ValueError(
                f"Task {row['task_id']} requires {resource_class}, but that worker pool is disabled"
            )
        if required_cpus > available_cpus or required_memory > available_memory:
            raise ValueError(
                f"Task {row['task_id']} requests {required_cpus} CPUs/{required_memory} GB, "
                f"exceeding the {resource_class} worker capacity of "
                f"{available_cpus} CPUs/{available_memory} GB"
            )


def _write_pool_script(
    path: Path,
    resource_class: str,
    workers: int,
    partition: str,
    cpus: int,
    memory_gb: int,
    database: Path,
    work_root: Path,
    log_dir: Path,
    gpu: bool,
) -> None:
    directives = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name=neuroprep-{resource_class}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --array=0-{workers - 1}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={memory_gb}G",
        "#SBATCH --time=7-00:00:00",
        f"#SBATCH --output={log_dir}/%x-%A_%a.out",
    ]
    if gpu:
        directives.append("#SBATCH --gres=gpu:1")
    command = (
        '"${NEUROPREP_PYTHON:-python3}" -m neuroprep.cli worker '
        f"--database {shlex.quote(str(database.resolve()))} "
        f"--work-root {shlex.quote(str(work_root.resolve()))} "
        f"--resource-class {resource_class} "
        '--worker-id "${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"'
    )
    text = "\n".join([*directives, "", "set -euo pipefail", command, ""])
    _atomic_write(path, text)


def _atomic_write(path: Path, text: str, executable: bool = False) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    if executable:
        temporary.chmod(0o750)
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
