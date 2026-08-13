from __future__ import annotations

import json
from pathlib import Path

from .execution_models import ExecutionPlan
from .settings import validated_cluster_path
from .state import RunState


def compile_backend_evidence(plan: ExecutionPlan, state: RunState) -> dict[str, str]:
    """Convert a terminal representative run into selector-compatible evidence.

    Evidence is emitted only for smoke specifications whose executor actually
    ran. Incomplete runs are rejected; terminal failures are recorded as failed;
    passes require every recorded artifact to remain present.
    """

    records = state.task_records()
    by_executor: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_executor.setdefault(str(record["executor_id"]), []).append(record)

    evidence: dict[str, str] = {}
    for smoke in plan.planned_smoke_tests:
        test_id = str(smoke["test_id"])
        executor_id = str(smoke["executor_id"])
        relevant = by_executor.get(executor_id, [])
        if not relevant:
            continue
        statuses = {str(record["status"]) for record in relevant}
        unfinished = statuses & {"pending", "running"}
        if unfinished:
            raise ValueError(
                f"qualification evidence requested before {executor_id} is terminal: "
                + ", ".join(sorted(unfinished))
            )
        if statuses & {"failed", "blocked"}:
            evidence[test_id] = "failed"
            continue
        if statuses <= {"complete", "skipped"} and "complete" in statuses:
            missing: list[str] = []
            for record in relevant:
                if record["status"] != "complete":
                    continue
                raw = record.get("artifacts_json") or "[]"
                for value in json.loads(str(raw)):
                    path = validated_cluster_path(str(value))
                    if not Path(path).exists():
                        missing.append(str(path))
            evidence[test_id] = "failed" if missing else "passed"
    return dict(sorted(evidence.items()))
