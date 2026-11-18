from __future__ import annotations

import glob
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .execution_models import ExecutableTask, ExecutionPlan
from .settings import validated_cluster_path


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _aware_timestamp(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a UTC offset")
    return parsed.astimezone(UTC)


class RunState:
    def __init__(self, path: str) -> None:
        self.path = validated_cluster_path(path)

    @contextmanager
    def connect(self, *, read_only: bool = False):
        if read_only:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            uri = f"{self.path.as_uri()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=60)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=60)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=60000")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self, plan: ExecutionPlan, plan_path: str) -> None:
        validated_cluster_path(plan_path, must_exist=True)
        with self.connect() as connection:
            self._schema(connection)
            existing = connection.execute(
                "SELECT value FROM metadata WHERE key='catalog_digest'"
            ).fetchone()
            if existing and existing["value"] != plan.catalog_digest:
                raise ValueError("state database catalog digest differs from the execution plan")
            metadata = {
                "dataset_id": plan.dataset_id,
                "catalog_digest": plan.catalog_digest,
                "plan_path": plan_path,
                "initialized_at": _now(),
            }
            connection.executemany(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
                metadata.items(),
            )
            observed_ids = {
                row["task_id"] for row in connection.execute("SELECT task_id FROM tasks")
            }
            planned_ids = {task.task_id for task in plan.tasks}
            unexpected = observed_ids - planned_ids
            if unexpected:
                raise ValueError(
                    "state database contains tasks absent from the plan: "
                    + ", ".join(sorted(unexpected)[:5])
                )
            for task in plan.tasks:
                row = connection.execute(
                    "SELECT fingerprint FROM tasks WHERE task_id=?", (task.task_id,)
                ).fetchone()
                if row and row["fingerprint"] != task.fingerprint:
                    raise ValueError(f"task fingerprint changed: {task.task_id}")
                if row:
                    continue
                status = "pending" if plan.runs(task) else "skipped"
                connection.execute(
                    """
                    INSERT INTO tasks(
                        task_id,status,executor_id,resource_class,required,fingerprint,
                        task_json,attempts,max_attempts,updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        task.task_id,
                        status,
                        task.executor_id,
                        task.resource_class,
                        int(task.required),
                        task.fingerprint,
                        task.model_dump_json(),
                        0,
                        3,
                        _now(),
                    ),
                )
                connection.executemany(
                    "INSERT INTO dependencies(task_id,depends_on) VALUES (?,?)",
                    ((task.task_id, dependency) for dependency in task.depends_on),
                )
            connection.commit()

    def claim(
        self,
        worker_id: str,
        resource_classes: set[str],
        lease_seconds: int = 3600,
    ) -> ExecutableTask | None:
        if not worker_id.strip():
            raise ValueError("worker_id cannot be empty")
        if not resource_classes:
            raise ValueError("at least one resource class is required")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._recover_expired(connection, resource_classes)
            self._block_failed_descendants(connection)
            placeholders = ",".join("?" for _ in resource_classes)
            row = connection.execute(
                f"""
                SELECT t.* FROM tasks t
                WHERE t.status='pending'
                  AND t.resource_class IN ({placeholders})
                  AND NOT EXISTS (
                    SELECT 1 FROM dependencies d
                    JOIN tasks parent ON parent.task_id=d.depends_on
                    WHERE d.task_id=t.task_id
                      AND parent.status NOT IN ('complete','skipped')
                  )
                ORDER BY t.required DESC,t.task_id LIMIT 1
                """,
                tuple(sorted(resource_classes)),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            lease = (datetime.now(UTC) + timedelta(seconds=lease_seconds)).isoformat()
            attempts = int(row["attempts"]) + 1
            connection.execute(
                """
                UPDATE tasks SET status='running',claimed_by=?,lease_expires_at=?,
                    attempts=?,updated_at=?,last_error=NULL
                WHERE task_id=? AND status='pending'
                """,
                (worker_id, lease, attempts, _now(), row["task_id"]),
            )
            connection.execute(
                "INSERT INTO attempts(task_id,attempt,worker_id,status,started_at) "
                "VALUES (?,?,?,?,?)",
                (row["task_id"], attempts, worker_id, "running", _now()),
            )
            connection.commit()
            return ExecutableTask.model_validate_json(row["task_json"])

    def recover_stale_task(
        self,
        task_id: str,
        *,
        expected_worker_id: str,
        expected_attempt: int,
        expected_lease_expires_at: str,
        finished_at: str,
        reason: str,
        log_path: str,
        return_code: int | None = None,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Dry-run or recover one exact expired task using compare-and-swap guards."""

        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if not task_id.strip() or not expected_worker_id.strip():
            raise ValueError("task_id and expected_worker_id cannot be empty")
        if expected_attempt < 1:
            raise ValueError("expected_attempt must be positive")
        if not reason.strip():
            raise ValueError("reason cannot be empty")
        now = datetime.now(UTC)
        lease = _aware_timestamp(expected_lease_expires_at, field="expected_lease_expires_at")
        finished = _aware_timestamp(finished_at, field="finished_at")
        if lease >= now:
            raise ValueError("expected lease has not expired")
        if finished > now:
            raise ValueError("finished_at cannot be in the future")
        candidate = validated_cluster_path(log_path, must_exist=True)
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise ValueError("log_path must be a non-empty regular file")
        validated_log = str(candidate)

        with self.connect(read_only=not apply) as connection:
            connection.execute("BEGIN IMMEDIATE" if apply else "BEGIN")
            try:
                task_row = connection.execute(
                    """
                    SELECT task_id,status,executor_id,resource_class,attempts,max_attempts,
                           claimed_by,lease_expires_at,return_code,updated_at,last_error,
                           log_path
                    FROM tasks WHERE task_id=?
                    """,
                    (task_id,),
                ).fetchone()
                if task_row is None:
                    raise ValueError(f"unknown task: {task_id}")
                task = dict(task_row)
                if task["status"] != "running":
                    raise RuntimeError("task is not running; stale recovery CAS rejected")
                if task["claimed_by"] != expected_worker_id:
                    raise RuntimeError("task worker differs from expected_worker_id")
                if task["attempts"] != expected_attempt:
                    raise RuntimeError("task attempt differs from expected_attempt")
                if task["lease_expires_at"] != expected_lease_expires_at:
                    raise RuntimeError("task lease differs from expected_lease_expires_at")
                if task["attempts"] >= task["max_attempts"]:
                    raise RuntimeError(
                        "task has no retry remaining; stale recovery only requeues retryable tasks"
                    )

                attempt_row = connection.execute(
                    "SELECT * FROM attempts WHERE task_id=? AND attempt=?",
                    (task_id, expected_attempt),
                ).fetchone()
                if attempt_row is None:
                    raise RuntimeError("matching attempt row is absent")
                attempt = dict(attempt_row)
                if attempt["worker_id"] != expected_worker_id or attempt["status"] != "running":
                    raise RuntimeError("attempt ownership/status differs from expected state")
                if any(
                    attempt[field] is not None
                    for field in ("finished_at", "return_code", "error", "log_path")
                ):
                    raise RuntimeError("attempt already contains terminal evidence")
                started = _aware_timestamp(attempt["started_at"], field="attempt.started_at")
                if finished < started:
                    raise ValueError("finished_at precedes the attempt start")

                report: dict[str, Any] = {
                    "action": "recover_stale_task",
                    "mode": "apply" if apply else "dry-run",
                    "applied": apply,
                    "task_before": task,
                    "attempt_before": attempt,
                    "transition": {
                        "status": "pending",
                        "claimed_by": None,
                        "lease_expires_at": None,
                        "return_code": return_code,
                        "last_error": reason,
                        "log_path": validated_log,
                        "attempt_status": "failed",
                        "attempt_finished_at": finished_at,
                    },
                }
                if not apply:
                    return report

                applied_at = _now()
                task_cursor = connection.execute(
                    """
                    UPDATE tasks
                    SET status='pending',claimed_by=NULL,lease_expires_at=NULL,
                        return_code=?,updated_at=?,last_error=?,log_path=?
                    WHERE task_id=? AND status='running' AND claimed_by=?
                      AND attempts=? AND lease_expires_at=?
                    """,
                    (
                        return_code,
                        applied_at,
                        reason,
                        validated_log,
                        task_id,
                        expected_worker_id,
                        expected_attempt,
                        expected_lease_expires_at,
                    ),
                )
                if task_cursor.rowcount != 1:
                    raise RuntimeError("task recovery compare-and-swap failed")
                attempt_cursor = connection.execute(
                    """
                    UPDATE attempts
                    SET status='failed',finished_at=?,return_code=?,error=?,log_path=?
                    WHERE task_id=? AND attempt=? AND worker_id=? AND status='running'
                      AND finished_at IS NULL AND return_code IS NULL
                      AND error IS NULL AND log_path IS NULL
                    """,
                    (
                        finished_at,
                        return_code,
                        reason,
                        validated_log,
                        task_id,
                        expected_attempt,
                        expected_worker_id,
                    ),
                )
                if attempt_cursor.rowcount != 1:
                    raise RuntimeError("attempt recovery compare-and-swap failed")
                task_after = dict(
                    connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
                )
                attempt_after = dict(
                    connection.execute(
                        "SELECT * FROM attempts WHERE task_id=? AND attempt=?",
                        (task_id, expected_attempt),
                    ).fetchone()
                )
                connection.commit()
                report["task_after"] = task_after
                report["attempt_after"] = attempt_after
                return report
            except Exception:
                if apply:
                    connection.rollback()
                raise

    def reconcile_blocked_children(
        self,
        child_parent_pairs: dict[str, str],
        *,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Dry-run or release exact dependency-failed children with verified parents."""

        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if not child_parent_pairs:
            raise ValueError("at least one child-parent pair is required")
        if any(
            not child.strip() or not parent.strip() for child, parent in child_parent_pairs.items()
        ):
            raise ValueError("child and parent task ids cannot be empty")

        with self.connect(read_only=not apply) as connection:
            connection.execute("BEGIN IMMEDIATE" if apply else "BEGIN")
            try:
                candidates: list[dict[str, Any]] = []
                for child_id, parent_id in sorted(child_parent_pairs.items()):
                    child_row = connection.execute(
                        """
                        SELECT task_id,status,executor_id,resource_class,attempts,max_attempts,
                               claimed_by,lease_expires_at,return_code,updated_at,last_error,
                               log_path,artifacts_json
                        FROM tasks WHERE task_id=?
                        """,
                        (child_id,),
                    ).fetchone()
                    if child_row is None:
                        raise ValueError(f"unknown child task: {child_id}")
                    child = dict(child_row)
                    if (
                        child["status"] != "blocked"
                        or child["last_error"] != "dependency failed"
                        or child["attempts"] != 0
                        or child["claimed_by"] is not None
                        or child["lease_expires_at"] is not None
                        or child["return_code"] is not None
                        or child["log_path"] is not None
                        or child["artifacts_json"] is not None
                    ):
                        raise RuntimeError(
                            f"child is not an unattempted dependency-failed row: {child_id}"
                        )

                    parent_row = connection.execute(
                        """
                        SELECT task_id,status,executor_id,resource_class,attempts,max_attempts,
                               claimed_by,lease_expires_at,return_code,updated_at,last_error,
                               log_path,artifacts_json
                        FROM tasks WHERE task_id=?
                        """,
                        (parent_id,),
                    ).fetchone()
                    if parent_row is None:
                        raise ValueError(f"unknown repaired parent task: {parent_id}")
                    parent = dict(parent_row)
                    if (
                        parent["status"] != "complete"
                        or parent["return_code"] != 0
                        or parent["claimed_by"] is not None
                        or parent["lease_expires_at"] is not None
                    ):
                        raise RuntimeError(f"repaired parent is not complete: {parent_id}")
                    relation = connection.execute(
                        "SELECT 1 FROM dependencies WHERE task_id=? AND depends_on=?",
                        (child_id, parent_id),
                    ).fetchone()
                    if relation is None:
                        raise RuntimeError(
                            f"expected repaired parent is not a dependency of child: {child_id}"
                        )
                    dependencies = [
                        dict(row)
                        for row in connection.execute(
                            """
                            SELECT d.depends_on,p.status,p.executor_id,p.resource_class,
                                   p.attempts,p.max_attempts,p.claimed_by,p.lease_expires_at,
                                   p.return_code,p.updated_at,p.last_error,p.log_path,
                                   p.artifacts_json,p.fingerprint,p.task_json
                            FROM dependencies d JOIN tasks p ON p.task_id=d.depends_on
                            WHERE d.task_id=? ORDER BY d.depends_on
                            """,
                            (child_id,),
                        )
                    ]
                    if not dependencies or any(
                        dependency["status"] not in {"complete", "skipped"}
                        for dependency in dependencies
                    ):
                        raise RuntimeError(f"child still has an incomplete dependency: {child_id}")
                    artifact_ledgers: dict[str, list[str]] = {}
                    completion_attempts: dict[str, dict[str, Any]] = {}
                    for dependency in dependencies:
                        dependency_id = dependency["depends_on"]
                        if dependency["status"] == "skipped":
                            if (
                                dependency["attempts"] != 0
                                or dependency["claimed_by"] is not None
                                or dependency["lease_expires_at"] is not None
                                or dependency["return_code"] is not None
                                or dependency["last_error"] is not None
                                or dependency["log_path"] is not None
                                or dependency["artifacts_json"] is not None
                            ):
                                raise RuntimeError(
                                    f"skipped dependency state is inconsistent: {dependency_id}"
                                )
                            artifact_ledgers[dependency_id] = []
                            continue
                        if (
                            dependency["return_code"] != 0
                            or dependency["claimed_by"] is not None
                            or dependency["lease_expires_at"] is not None
                            or dependency["last_error"] is not None
                        ):
                            raise RuntimeError(
                                f"complete dependency state is inconsistent: {dependency_id}"
                            )
                        completion_attempts[dependency_id] = self._validated_completion_attempt(
                            connection,
                            dependency,
                        )
                        artifact_ledgers[dependency_id] = self._validated_recorded_artifacts(
                            dependency
                        )
                    candidates.append(
                        {
                            "child_before": child,
                            "repaired_parent": {
                                key: parent[key]
                                for key in (
                                    "task_id",
                                    "status",
                                    "executor_id",
                                    "attempts",
                                    "return_code",
                                    "updated_at",
                                    "log_path",
                                )
                            },
                            "dependency_states": [
                                {
                                    key: dependency[key]
                                    for key in (
                                        "depends_on",
                                        "status",
                                        "executor_id",
                                        "attempts",
                                        "return_code",
                                        "updated_at",
                                        "log_path",
                                    )
                                }
                                for dependency in dependencies
                            ],
                            "validated_completion_attempts": completion_attempts,
                            "validated_dependency_artifacts": artifact_ledgers,
                        }
                    )

                report: dict[str, Any] = {
                    "action": "reconcile_blocked_children",
                    "mode": "apply" if apply else "dry-run",
                    "applied": apply,
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                }
                if not apply:
                    return report

                applied_at = _now()
                for child_id, parent_id in sorted(child_parent_pairs.items()):
                    cursor = connection.execute(
                        """
                        UPDATE tasks SET status='pending',last_error=NULL,updated_at=?
                        WHERE task_id=? AND status='blocked'
                          AND last_error='dependency failed' AND attempts=0
                          AND claimed_by IS NULL AND lease_expires_at IS NULL
                          AND return_code IS NULL AND log_path IS NULL
                          AND artifacts_json IS NULL
                          AND EXISTS (
                            SELECT 1 FROM dependencies d JOIN tasks p ON p.task_id=d.depends_on
                            WHERE d.task_id=tasks.task_id AND d.depends_on=?
                              AND p.status='complete' AND p.return_code=0
                          )
                          AND NOT EXISTS (
                            SELECT 1 FROM dependencies d JOIN tasks p ON p.task_id=d.depends_on
                            WHERE d.task_id=tasks.task_id
                              AND (
                                p.status NOT IN ('complete','skipped')
                                OR (p.status='complete' AND (
                                  p.return_code IS NULL OR p.return_code!=0
                                ))
                              )
                          )
                        """,
                        (applied_at, child_id, parent_id),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError(
                            f"child reconciliation compare-and-swap failed: {child_id}"
                        )
                children_after = [
                    dict(row)
                    for row in connection.execute(
                        "SELECT * FROM tasks WHERE task_id IN "
                        f"({','.join('?' for _ in child_parent_pairs)}) "
                        "ORDER BY task_id",
                        tuple(sorted(child_parent_pairs)),
                    )
                ]
                connection.commit()
                report["children_after"] = children_after
                return report
            except Exception:
                if apply:
                    connection.rollback()
                raise

    @staticmethod
    def _validated_completion_attempt(
        connection: sqlite3.Connection,
        task: dict[str, Any],
    ) -> dict[str, Any]:
        attempt_number = task["attempts"]
        if not isinstance(attempt_number, int) or attempt_number < 1:
            raise RuntimeError(f"complete dependency has no recorded attempt: {task['depends_on']}")
        row = connection.execute(
            "SELECT * FROM attempts WHERE task_id=? AND attempt=?",
            (task["depends_on"], attempt_number),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"complete dependency has no matching attempt: {task['depends_on']}")
        attempt = dict(row)
        if (
            attempt["status"] != "complete"
            or attempt["return_code"] != 0
            or attempt["finished_at"] is None
            or attempt["error"] is not None
            or attempt["log_path"] != task["log_path"]
        ):
            raise RuntimeError(
                f"complete dependency lacks a matching successful attempt: {task['depends_on']}"
            )
        _aware_timestamp(attempt["finished_at"], field="attempt.finished_at")
        if not task["log_path"]:
            raise RuntimeError(f"complete dependency log is absent: {task['depends_on']}")
        log_path = validated_cluster_path(task["log_path"], must_exist=True)
        if not log_path.is_file() or log_path.stat().st_size == 0:
            raise RuntimeError(f"complete dependency log is empty: {task['depends_on']}")
        return attempt

    @staticmethod
    def _validated_recorded_artifacts(parent: dict[str, Any]) -> list[str]:
        try:
            artifacts = json.loads(parent["artifacts_json"] or "null")
        except json.JSONDecodeError as exc:
            raise RuntimeError("dependency artifact ledger is invalid JSON") from exc
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or any(not isinstance(path, str) or not path for path in artifacts)
        ):
            raise RuntimeError("dependency has no valid artifact ledger")
        validated: list[str] = []
        for raw in artifacts:
            path = validated_cluster_path(raw, must_exist=True)
            if not RunState._artifact_nonempty(path):
                raise RuntimeError(f"dependency artifact is empty: {path}")
            validated.append(str(path))
        task = ExecutableTask.model_validate_json(parent["task_json"])
        if task.task_id != parent["depends_on"] or task.fingerprint != parent["fingerprint"]:
            raise RuntimeError(f"dependency task record is inconsistent: {parent['depends_on']}")
        matched: set[str] = set()
        for requirement in task.artifacts:
            if requirement.validator != "exists_nonempty":
                raise RuntimeError("dependency uses an unsupported artifact validator")
            requirement_matches: set[str] = set()
            for pattern in requirement.patterns:
                candidate = Path(pattern)
                expanded = str(
                    candidate if candidate.is_absolute() else Path(task.output_root) / candidate
                )
                for raw_path in glob.glob(expanded, recursive=True):
                    path = Path(raw_path)
                    if RunState._artifact_nonempty(path):
                        requirement_matches.add(str(path))
            if requirement.required and len(requirement_matches) < requirement.minimum_count:
                raise RuntimeError(f"dependency artifact contract failed: {parent['depends_on']}")
            matched.update(requirement_matches)
        if not matched or not matched.issubset(set(validated)):
            raise RuntimeError(
                f"dependency artifacts disagree with its plan: {parent['depends_on']}"
            )
        return sorted(validated)

    @staticmethod
    def _artifact_nonempty(path: Path) -> bool:
        if path.is_file():
            return path.stat().st_size > 0
        if path.is_dir():
            return any(
                item.is_file() and item.stat().st_size > 0
                for item in path.rglob("*")
            )
        return False

    def renew(self, task_id: str, worker_id: str, lease_seconds: int = 3600) -> None:
        lease = (datetime.now(UTC) + timedelta(seconds=lease_seconds)).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE tasks SET lease_expires_at=?,updated_at=?
                WHERE task_id=? AND status='running' AND claimed_by=?
                """,
                (lease, _now(), task_id, worker_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"cannot renew unowned task {task_id}")
            connection.commit()

    def finish(
        self,
        task_id: str,
        worker_id: str,
        *,
        return_code: int,
        log_path: str,
        error: str | None = None,
        artifacts: list[str] | None = None,
    ) -> None:
        status = "complete" if return_code == 0 and not error else "failed"
        validated_cluster_path(log_path)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT attempts,max_attempts FROM tasks WHERE task_id=? AND claimed_by=?",
                (task_id, worker_id),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"worker does not own task {task_id}")
            attempts = int(row["attempts"])
            final_status = status
            if status == "failed" and attempts < int(row["max_attempts"]):
                final_status = "pending"
            connection.execute(
                """
                UPDATE tasks SET status=?,return_code=?,claimed_by=NULL,lease_expires_at=NULL,
                    updated_at=?,last_error=?,log_path=?,artifacts_json=? WHERE task_id=?
                """,
                (
                    final_status,
                    return_code,
                    _now(),
                    error,
                    log_path,
                    json.dumps(artifacts or []),
                    task_id,
                ),
            )
            connection.execute(
                """
                UPDATE attempts SET status=?,finished_at=?,return_code=?,error=?,log_path=?
                WHERE task_id=? AND attempt=?
                """,
                (status, _now(), return_code, error, log_path, task_id, attempts),
            )
            if final_status == "failed":
                self._block_failed_descendants(connection)
            connection.commit()

    def summary(self) -> dict[str, Any]:
        with self.connect() as connection:
            self._schema(connection)
            counts = {
                row["status"]: row["count"]
                for row in connection.execute(
                    "SELECT status,count(*) AS count FROM tasks GROUP BY status"
                )
            }
            active = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT task_id,status,executor_id,resource_class,attempts,claimed_by,
                           lease_expires_at,last_error,log_path
                    FROM tasks WHERE status!='complete' ORDER BY status,task_id
                    """
                )
            ]
            return {"counts": counts, "active": active, "total": sum(counts.values())}

    def task_records(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            self._schema(connection)
            return [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT task_id,status,executor_id,resource_class,required,fingerprint,
                           task_json,attempts,return_code,updated_at,last_error,log_path,
                           artifacts_json
                    FROM tasks ORDER BY task_id
                    """
                )
            ]

    def has_unfinished_work(self, resource_classes: set[str]) -> bool:
        if not resource_classes:
            return False
        placeholders = ",".join("?" for _ in resource_classes)
        with self.connect() as connection:
            self._schema(connection)
            row = connection.execute(
                f"""
                SELECT 1 FROM tasks
                WHERE status IN ('pending','running')
                  AND resource_class IN ({placeholders}) LIMIT 1
                """,
                tuple(sorted(resource_classes)),
            ).fetchone()
            return row is not None

    def _recover_expired(
        self,
        connection: sqlite3.Connection,
        resource_classes: set[str],
    ) -> None:
        now = _now()
        placeholders = ",".join("?" for _ in resource_classes)
        rows = connection.execute(
            f"SELECT task_id,attempts,max_attempts FROM tasks "
            f"WHERE status='running' AND lease_expires_at<? "
            f"AND resource_class IN ({placeholders})",
            (now, *sorted(resource_classes)),
        ).fetchall()
        for row in rows:
            status = "pending" if row["attempts"] < row["max_attempts"] else "failed"
            connection.execute(
                """
                UPDATE tasks SET status=?,claimed_by=NULL,lease_expires_at=NULL,
                    last_error='worker lease expired',updated_at=? WHERE task_id=?
                """,
                (status, now, row["task_id"]),
            )
            connection.execute(
                """
                UPDATE attempts SET status='failed',finished_at=?,error='worker lease expired'
                WHERE task_id=? AND attempt=? AND status='running'
                """,
                (now, row["task_id"], row["attempts"]),
            )

    def _block_failed_descendants(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            UPDATE tasks SET status='blocked',updated_at=?,last_error='dependency failed'
            WHERE status='pending' AND EXISTS (
                SELECT 1 FROM dependencies d JOIN tasks parent ON parent.task_id=d.depends_on
                WHERE d.task_id=tasks.task_id AND parent.status IN ('failed','blocked')
            )
            """,
            (_now(),),
        )

    @staticmethod
    def _schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tasks(
                task_id TEXT PRIMARY KEY,status TEXT NOT NULL,executor_id TEXT NOT NULL,
                resource_class TEXT NOT NULL,required INTEGER NOT NULL,fingerprint TEXT NOT NULL,
                task_json TEXT NOT NULL,attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,claimed_by TEXT,lease_expires_at TEXT,
                return_code INTEGER,updated_at TEXT NOT NULL,last_error TEXT,log_path TEXT,
                artifacts_json TEXT
            );
            CREATE TABLE IF NOT EXISTS dependencies(
                task_id TEXT NOT NULL REFERENCES tasks(task_id),
                depends_on TEXT NOT NULL REFERENCES tasks(task_id),
                PRIMARY KEY(task_id,depends_on)
            );
            CREATE TABLE IF NOT EXISTS attempts(
                task_id TEXT NOT NULL REFERENCES tasks(task_id),attempt INTEGER NOT NULL,
                worker_id TEXT NOT NULL,status TEXT NOT NULL,started_at TEXT NOT NULL,
                finished_at TEXT,return_code INTEGER,error TEXT,log_path TEXT,
                PRIMARY KEY(task_id,attempt)
            );
            CREATE INDEX IF NOT EXISTS tasks_ready ON tasks(status,resource_class,required,task_id);
            CREATE INDEX IF NOT EXISTS deps_child ON dependencies(task_id);
            """
        )
