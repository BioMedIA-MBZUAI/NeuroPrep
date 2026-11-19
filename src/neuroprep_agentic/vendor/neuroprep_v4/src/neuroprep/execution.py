from __future__ import annotations

import json
import base64
import csv
import shlex
import subprocess
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_NEXTFLOW_STAGES = (
    "ingest",
    "validate",
    "preprocess",
    "denoise",
    "postprocess",
    "register",
    "qc",
    "release",
    "index",
    "grid",
    "assemble",
    "align",
    "tensor",
    "link",
    "split",
    "harmonize",
    "audit",
)


@dataclass
class ExecutionResult:
    task_id: str
    status: str
    return_code: int | None = None
    message: str | None = None


def load_plan(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload.get("tasks"), list):
        raise ValueError("Plan is missing a tasks list")
    return payload


def topological_tasks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = {task["task_id"]: task for task in plan["tasks"]}
    indegree = {task_id: 0 for task_id in tasks}
    dependents: dict[str, list[str]] = defaultdict(list)
    for task_id, task in tasks.items():
        for dependency in task.get("dependencies", []):
            if dependency not in tasks:
                raise ValueError(f"Unknown dependency {dependency} for task {task_id}")
            indegree[task_id] += 1
            dependents[dependency].append(task_id)

    queue = deque(sorted(task_id for task_id, degree in indegree.items() if degree == 0))
    ordered: list[dict[str, Any]] = []
    while queue:
        task_id = queue.popleft()
        ordered.append(tasks[task_id])
        for dependent in sorted(dependents[task_id]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                queue.append(dependent)
    if len(ordered) != len(tasks):
        raise ValueError("Plan contains a dependency cycle")
    return ordered


def render_commands(plan: dict[str, Any]) -> list[str]:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for task in topological_tasks(plan):
        command = task.get("command") or []
        lines.append(f"# {task['task_id']} [{task['stage']}] {task['pipeline']}")
        if command:
            lines.append(shlex.join(str(part) for part in command))
        else:
            lines.append(f"# BLOCKED: custom command must be configured for {task['pipeline']}")
        lines.append("")
    return lines


def write_nextflow_tasks(
    plan: dict[str, Any],
    path: Path,
    allow_custom_gaps: bool = False,
) -> int:
    """Write tasks for the generic stage-barrier Nextflow workflow.

    Empty commands represent dataset mappings or modality pipelines that have not
    yet been configured. Refuse to create a partially runnable task table unless
    the caller explicitly opts in; downstream validation must never accidentally
    run against an incomplete BIDS dataset.
    """

    ordered = topological_tasks(plan)
    gaps = [task for task in ordered if not (task.get("command") or [])]
    if gaps and not allow_custom_gaps:
        examples = ", ".join(task["task_id"] for task in gaps[:3])
        remainder = f" (+{len(gaps) - 3} more)" if len(gaps) > 3 else ""
        raise ValueError(
            "Plan contains tasks without configured commands: "
            f"{examples}{remainder}. Configure them or pass --allow-custom-gaps."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("task_id", "stage", "subject", "cpus", "memory_gb", "time", "command_b64"),
            delimiter="\t",
        )
        writer.writeheader()
        present_stages: set[str] = set()
        for task in ordered:
            command = [str(value) for value in (task.get("command") or [])]
            if not command:
                continue
            resources = task.get("resources", {})
            encoded = base64.b64encode(shlex.join(command).encode("utf-8")).decode("ascii")
            writer.writerow(
                {
                    "task_id": task["task_id"],
                    "stage": task["stage"],
                    "subject": task.get("subject") or "dataset",
                    "cpus": int(resources.get("cpus", 1)),
                    "memory_gb": int(resources.get("memory_gb", 4)),
                    "time": resources.get("time", "04:00:00"),
                    "command_b64": encoded,
                }
            )
            present_stages.add(str(task["stage"]))
            count += 1

        # The Nextflow graph uses one stage-completion channel as a barrier for
        # the next stage. A cheap sentinel keeps that channel alive when a
        # modality-specific stage is legitimately absent from this dataset.
        for stage in _NEXTFLOW_STAGES:
            if stage in present_stages:
                continue
            writer.writerow(
                {
                    "task_id": f"__neuroprep_noop_{stage}",
                    "stage": stage,
                    "subject": "dataset",
                    "cpus": 1,
                    "memory_gb": 1,
                    "time": "00:05:00",
                    "command_b64": base64.b64encode(b"true").decode("ascii"),
                }
            )
    return count


def execute_plan(
    plan: dict[str, Any],
    state_path: Path,
    dry_run: bool = True,
    allow_custom_gaps: bool = False,
) -> list[ExecutionResult]:
    state = _load_state(state_path)
    results: list[ExecutionResult] = []
    for task in topological_tasks(plan):
        task_id = task["task_id"]
        if state.get(task_id, {}).get("status") == "complete":
            results.append(ExecutionResult(task_id, "skipped", message="already complete"))
            continue
        command = [str(value) for value in (task.get("command") or [])]
        if not command:
            result = ExecutionResult(task_id, "blocked", message="custom command is not configured")
            results.append(result)
            if not allow_custom_gaps:
                break
            continue
        if dry_run:
            results.append(ExecutionResult(task_id, "dry-run", message=shlex.join(command)))
            continue

        _update_state(state_path, state, task_id, "running")
        completed = subprocess.run(command, check=False)
        status = "complete" if completed.returncode == 0 else "failed"
        _update_state(state_path, state, task_id, status, completed.returncode)
        results.append(ExecutionResult(task_id, status, completed.returncode))
        if completed.returncode != 0:
            break
    return results


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload if isinstance(payload, dict) else {}


def _update_state(
    path: Path,
    state: dict[str, Any],
    task_id: str,
    status: str,
    return_code: int | None = None,
) -> None:
    state[task_id] = {
        "status": status,
        "return_code": return_code,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)
