from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from .execution_models import ExecutableTask, ExecutionPlan
from .settings import validated_cluster_path


def _now() -> str:
    return datetime.now(UTC).isoformat()


class RunState:
    def __init__(self, path: str) -> None:
        self.path = validated_cluster_path(path)

    @contextmanager
    def connect(self):
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
                status = "pending" if task.required or plan.layout.execute_optional else "skipped"
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
            self._recover_expired(connection)
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

    def _recover_expired(self, connection: sqlite3.Connection) -> None:
        now = _now()
        rows = connection.execute(
            "SELECT task_id,attempts,max_attempts FROM tasks "
            "WHERE status='running' AND lease_expires_at<?",
            (now,),
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
