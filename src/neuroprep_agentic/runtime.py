from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import TextIO

from .compiler import executable_task_fingerprint
from .execution_models import ArtifactRequirement, ExecutableTask
from .settings import validated_cluster_path
from .state import RunState


def _atomic_symlink(target: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".link-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(target)
    temporary.replace(destination)


def _prepare_primary_anatomical_bids_view(task: ExecutableTask) -> None:
    if task.executor_id not in {"smriprep", "deepprep"}:
        return
    primary_value = task.metadata.get("primary_anatomical_bids_input")
    if not isinstance(primary_value, str) or not primary_value:
        raise ValueError(
            f"{task.executor_id} task lacks a deterministic primary anatomical input"
        )
    primary = validated_cluster_path(primary_value, must_exist=True)
    bids_root_value = task.metadata.get("bids_root")
    if not isinstance(bids_root_value, str):
        # Older plans did not repeat the layout root in task metadata. Derive it
        # from the primary path and the first BIDS subject component.
        subject_component = next(
            (parent for parent in primary.parents if parent.name.startswith("sub-")), None
        )
        if subject_component is None:
            raise ValueError(f"Cannot derive BIDS root from {primary}")
        bids_root = subject_component.parent
    else:
        bids_root = validated_cluster_path(bids_root_value, must_exist=True)
    try:
        relative = primary.relative_to(bids_root)
    except ValueError as exc:
        raise ValueError(f"Primary anatomical input is outside BIDS root: {primary}") from exc

    view = validated_cluster_path(str(Path(task.work_dir) / f"{task.executor_id}-bids"))
    view.mkdir(parents=True, exist_ok=True)
    description_source = bids_root / "dataset_description.json"
    description_target = view / "dataset_description.json"
    if description_source.is_file():
        shutil.copy2(description_source, description_target)
    else:
        description_target.write_text(
            json.dumps(
                {
                    "Name": "NeuroPrep primary-anatomical view",
                    "BIDSVersion": "1.10.0",
                    "DatasetType": "raw",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    container_target = str(Path("/source-data") / relative)
    _atomic_symlink(container_target, view / relative)
    sidecar = primary.with_name(
        primary.name[:-7] + ".json" if primary.name.endswith(".nii.gz") else primary.stem + ".json"
    )
    if sidecar.is_file():
        sidecar_relative = sidecar.relative_to(bids_root)
        _atomic_symlink(str(Path("/source-data") / sidecar_relative), view / sidecar_relative)
    for name in ("participants.tsv", "participants.json"):
        if (bids_root / name).is_file():
            _atomic_symlink(str(Path("/source-data") / name), view / name)


def _prepare_deepprep_bids_view(task: ExecutableTask) -> None:
    """Compatibility shim for callers of the original narrow helper."""
    _prepare_primary_anatomical_bids_view(task)


def _task_environment(task: ExecutableTask) -> dict[str, str]:
    work_dir = validated_cluster_path(task.work_dir)
    temporary = work_dir / "tmp"
    var_tmp = work_dir / "var-tmp"
    xdg_cache = work_dir / "xdg-cache"
    xdg_config = work_dir / "xdg-config"
    xdg_data = work_dir / "xdg-data"
    xdg_state = work_dir / "xdg-state"
    mpl_config = work_dir / "mplconfig"
    numba_cache = work_dir / "numba-cache"
    singularity_tmp = work_dir / "singularity-tmp"
    for path in (
        work_dir,
        temporary,
        var_tmp,
        xdg_cache,
        xdg_config,
        xdg_data,
        xdg_state,
        mpl_config,
        numba_cache,
        singularity_tmp,
    ):
        path.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:" + environment.get("PATH", ""),
            "XDG_CACHE_HOME": str(xdg_cache),
            "XDG_CONFIG_HOME": str(xdg_config),
            "XDG_DATA_HOME": str(xdg_data),
            "XDG_STATE_HOME": str(xdg_state),
            "MPLCONFIGDIR": str(mpl_config),
            "NUMBA_CACHE_DIR": str(numba_cache),
            "MNE_DONTWRITE_HOME": "true",
            "TMPDIR": str(temporary),
            "TMP": str(temporary),
            "TEMP": str(temporary),
            "SINGULARITY_TMPDIR": str(singularity_tmp),
            "APPTAINER_TMPDIR": str(singularity_tmp),
            "SINGULARITYENV_TMPDIR": "/tmp",
            "SINGULARITYENV_TMP": "/tmp",
            "SINGULARITYENV_TEMP": "/tmp",
        }
    )
    environment.update(task.environment)
    if task.command and Path(task.command[0]).name in {"singularity", "apptainer"}:
        bind_mounts: list[tuple[Path, str]] = []
        for index, token in enumerate(task.command[:-1]):
            if token not in {"--bind", "-B"}:
                continue
            parts = task.command[index + 1].split(":", 2)
            if len(parts) >= 2 and Path(parts[0]).is_absolute():
                bind_mounts.append((Path(parts[0]), parts[1]))
        bind_mounts.sort(key=lambda item: len(str(item[0])), reverse=True)

        for key, value in task.environment.items():
            container_value = value
            if Path(value).is_absolute():
                for host_root, container_root in bind_mounts:
                    try:
                        relative = Path(value).relative_to(host_root)
                    except ValueError:
                        continue
                    container_value = str(Path(container_root) / relative)
                    break
            # --cleanenv deliberately removes ordinary host variables. Both
            # prefixes are populated so SingularityCE and Apptainer receive the
            # task-scoped paths inside the container namespace.
            environment[f"SINGULARITYENV_{key}"] = container_value
            environment[f"APPTAINERENV_{key}"] = container_value

        cuda_devices = environment.get("CUDA_VISIBLE_DEVICES")
        if cuda_devices:
            for prefix in ("SINGULARITYENV_", "APPTAINERENV_"):
                environment[f"{prefix}CUDA_VISIBLE_DEVICES"] = cuda_devices
                environment[f"{prefix}NVIDIA_VISIBLE_DEVICES"] = cuda_devices
        rocr_devices = environment.get("ROCR_VISIBLE_DEVICES")
        if rocr_devices:
            for prefix in ("SINGULARITYENV_", "APPTAINERENV_"):
                environment[f"{prefix}ROCR_VISIBLE_DEVICES"] = rocr_devices
    return environment


def _matches(requirement: ArtifactRequirement, task: ExecutableTask) -> list[str]:
    matches: set[str] = set()
    for pattern in requirement.patterns:
        candidate = Path(pattern)
        expanded = pattern if candidate.is_absolute() else str(Path(task.output_root) / pattern)
        for match in glob.glob(expanded, recursive=True):
            path = Path(match)
            if (path.is_file() and path.stat().st_size > 0) or (
                path.is_dir() and any(path.iterdir())
            ):
                matches.add(str(path))
    return sorted(matches)


def validate_artifacts(task: ExecutableTask) -> tuple[list[str], list[str]]:
    observed: list[str] = []
    failures: list[str] = []
    for requirement in task.artifacts:
        matches = _matches(requirement, task)
        observed.extend(matches)
        if requirement.required and len(matches) < requirement.minimum_count:
            failures.append(
                f"{requirement.kind}: expected {requirement.minimum_count}, found {len(matches)}"
            )
    return sorted(set(observed)), failures


class LeaseRenewer:
    def __init__(self, state: RunState, task_id: str, worker_id: str, seconds: int) -> None:
        self.state = state
        self.task_id = task_id
        self.worker_id = worker_id
        self.seconds = seconds
        self.stop_event = threading.Event()
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> LeaseRenewer:
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.error:
            raise self.error

    def _run(self) -> None:
        interval = max(30, self.seconds // 3)
        while not self.stop_event.wait(interval):
            try:
                self.state.renew(self.task_id, self.worker_id, self.seconds)
            except Exception as exc:  # pragma: no cover - exercised by cluster integration
                self.error = exc
                self.stop_event.set()
                return


def execute_task(
    task: ExecutableTask,
    state: RunState,
    worker_id: str,
    *,
    lease_seconds: int = 3600,
) -> int:
    if task.fingerprint != executable_task_fingerprint(task):
        raise ValueError(f"task fingerprint verification failed: {task.task_id}")
    work_dir = validated_cluster_path(task.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    attempt_stamp = str(int(time.time() * 1000))
    log_path = work_dir / f"attempt-{attempt_stamp}.log"
    environment = _task_environment(task)
    _prepare_primary_anatomical_bids_view(task)
    error: str | None = None
    return_code = 1
    artifacts: list[str] = []
    try:
        with log_path.open("w", encoding="utf-8") as log:
            _write_header(log, task, worker_id)
            with LeaseRenewer(state, task.task_id, worker_id, lease_seconds):
                completed = subprocess.run(
                    task.command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    cwd=work_dir,
                    check=False,
                )
            return_code = completed.returncode
            if return_code == 0:
                artifacts, failures = validate_artifacts(task)
                if failures:
                    error = "artifact validation failed: " + "; ".join(failures)
                    return_code = 2
                    log.write(error + "\n")
    except Exception:
        error = traceback.format_exc(limit=40)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(error)
        return_code = 1
    state.finish(
        task.task_id,
        worker_id,
        return_code=return_code,
        log_path=str(log_path),
        error=error,
        artifacts=artifacts,
    )
    return return_code


def _write_header(log: TextIO, task: ExecutableTask, worker_id: str) -> None:
    payload = {
        "task_id": task.task_id,
        "worker_id": worker_id,
        "executor_id": task.executor_id,
        "fingerprint": task.fingerprint,
        "command": task.command,
    }
    log.write(json.dumps(payload, sort_keys=True) + "\n")
    log.flush()


def run_worker(
    state_path: str,
    worker_id: str,
    resource_classes: set[str],
    *,
    lease_seconds: int = 3600,
    max_tasks: int | None = None,
    idle_seconds: int = 30,
    exit_when_idle: bool = True,
) -> dict[str, int]:
    state = RunState(state_path)
    claimed = 0
    completed = 0
    failed = 0
    while max_tasks is None or claimed < max_tasks:
        task = state.claim(worker_id, resource_classes, lease_seconds)
        if task is None:
            if exit_when_idle and not state.has_unfinished_work(resource_classes):
                break
            time.sleep(idle_seconds)
            continue
        claimed += 1
        return_code = execute_task(task, state, worker_id, lease_seconds=lease_seconds)
        if return_code == 0:
            completed += 1
        else:
            failed += 1
    return {"claimed": claimed, "completed": completed, "failed": failed}
