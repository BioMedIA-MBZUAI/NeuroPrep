from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def task_fingerprint(task: Mapping[str, Any]) -> str:
    """Return the scientific identity of a task, excluding scheduling resources.

    CPU count, memory, walltime, attempt number, and timestamps may change without
    changing the derivative. Inputs, parameters, command/container, dependencies,
    and expected outputs may not.
    """

    identity = {
        key: task.get(key)
        for key in (
            "task_id",
            "dataset_id",
            "stage",
            "pipeline",
            "subject",
            "session",
            "dependencies",
            "inputs",
            "outputs",
            "params",
            "command",
        )
    }
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
