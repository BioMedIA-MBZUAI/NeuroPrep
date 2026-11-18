from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import paths

WorkflowName = Literal["coverage", "ingest_plan", "execution_review"]

# Home, NEUROPREP_HOME and NEUROPREP_ALLOWED_ROOTS (e.g. /nfs-stor/<user>:/scratch/<user>).
# Resolved once at import so a running plan cannot widen its own storage allowlist.
ALLOWED_ROOTS = paths.allowed_roots()
DEFAULT_AGENTIC_PROJECT = str(paths.default_agentic_project())


def validated_cluster_path(value: str, *, must_exist: bool = False) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Cluster paths must be absolute and cannot contain '..'")
    if not any(path == root or root in path.parents for root in ALLOWED_ROOTS):
        roots = ", ".join(str(root) for root in ALLOWED_ROOTS)
        raise ValueError(f"Path must be under one of: {roots}")
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def validated_agentic_project(value: str | None = None) -> Path:
    """Resolve the selected NeuroPrep-agentic repository and require its exact root.

    An explicit CLI/API value wins over ``NEUROPREP_AGENTIC_PROJECT``; the historical
    production path remains the compatibility default.  Returning the resolved git
    top-level prevents a plan from recording a symlink, subdirectory, or unrelated
    working tree while its commands execute code from somewhere else.
    """

    selected = (
        value
        if value is not None
        else os.getenv("NEUROPREP_AGENTIC_PROJECT", DEFAULT_AGENTIC_PROJECT)
    )
    candidate = validated_cluster_path(selected, must_exist=True).resolve(strict=True)
    if not candidate.is_dir():
        raise ValueError("agentic project must be an existing directory")
    try:
        completed = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("agentic project git repository cannot be inspected") from exc
    if completed.returncode != 0 or not completed.stdout.strip():
        raise ValueError("agentic project must be a git repository root")
    try:
        repository_root = Path(completed.stdout.strip()).resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise ValueError("agentic project git root cannot be resolved") from exc
    if repository_root != candidate:
        raise ValueError("agentic project must be the exact git repository root")
    return candidate


@dataclass(frozen=True)
class AgenticSettings:
    workflow: WorkflowName = "coverage"
    ollama_base_url: str = "http://127.0.0.1:11434"
    # Any ``provider/model`` understood by neuroprep_agentic.llm (ollama, openai, anthropic,
    # openrouter, vllm, lmstudio, openai_compatible).
    manager_model: str = "ollama/qwen3:14b"
    worker_model: str = "ollama/qwen3:8b"
    llm_base_url: str | None = None
    manifest_path: str | None = None
    execution_plan_path: str | None = None
    identity_key: str | None = None

    def __post_init__(self) -> None:
        if self.workflow not in {"coverage", "ingest_plan", "execution_review"}:
            raise ValueError(f"Unsupported workflow: {self.workflow}")
        if self.manifest_path:
            validated_cluster_path(self.manifest_path, must_exist=True)
        if self.execution_plan_path:
            validated_cluster_path(self.execution_plan_path, must_exist=True)
        if self.workflow == "ingest_plan":
            if not self.manifest_path:
                raise ValueError("ingest_plan requires NEUROPREP_MANIFEST_PATH")
            if not self.identity_key or len(self.identity_key) < 12:
                raise ValueError("ingest_plan requires a strong NEUROPREP_IDENTITY_KEY")
        if self.workflow == "execution_review" and not self.execution_plan_path:
            raise ValueError("execution_review requires NEUROPREP_EXECUTION_PLAN_PATH")
        from .llm import split_model_spec

        split_model_spec(self.manager_model)
        split_model_spec(self.worker_model)

    def endpoint(self, model: str):
        """Resolve ``model`` to a concrete endpoint, honoring this object's overrides."""

        from .llm import resolve_endpoint, split_model_spec

        provider, _ = split_model_spec(model)
        # A shared base URL is for self-hosted servers only; hosted providers use their own
        # *_BASE_URL variable so an API key is never sent to a local server's address.
        explicit = None if provider in {"openai", "openrouter", "anthropic"} else self.llm_base_url
        if provider == "ollama" and explicit is None and self.ollama_base_url:
            explicit = self.ollama_base_url
        return resolve_endpoint(model, base_url=explicit)

    @classmethod
    def from_env(cls, workflow: str | None = None) -> AgenticSettings:
        selected = workflow or os.getenv("NEUROPREP_WORKFLOW", "coverage")
        return cls(
            workflow=selected,  # type: ignore[arg-type]
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            manager_model=os.getenv("NEUROPREP_MANAGER_MODEL")
            or os.getenv("OLLAMA_MANAGER_MODEL", "ollama/qwen3:14b"),
            worker_model=os.getenv("NEUROPREP_WORKER_MODEL")
            or os.getenv("OLLAMA_WORKER_MODEL", "ollama/qwen3:8b"),
            llm_base_url=os.getenv("NEUROPREP_LLM_BASE_URL") or None,
            manifest_path=os.getenv("NEUROPREP_MANIFEST_PATH"),
            execution_plan_path=os.getenv("NEUROPREP_EXECUTION_PLAN_PATH"),
            identity_key=os.getenv("NEUROPREP_IDENTITY_KEY"),
        )
