from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from collections.abc import Collection
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import paths
from .backend_evidence import BackendEvidenceBinding
from .bids import bids_relative_path
from .execution_models import (
    ArtifactRequirement,
    ExecutableTask,
    ExecutionPlan,
    ExecutionScope,
    ExecutorBackend,
    ExecutorDefinition,
    RunLayout,
)
from .executors import ExecutorCatalog
from .models import CanonicalAcquisitionGraph, PlanResult, ResourceRequest, TaskSpec
from .structural import select_primary_t1


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def executable_task_fingerprint(task: ExecutableTask) -> str:
    payload = task.model_dump(mode="json")
    payload["fingerprint"] = "pending"
    return _fingerprint(payload)


def _scope_key(definition: ExecutorDefinition, task: TaskSpec) -> tuple[str, ...]:
    if definition.scope is ExecutionScope.DATASET:
        return (definition.executor_id, "dataset")
    if definition.scope is ExecutionScope.SUBJECT:
        return (definition.executor_id, task.subject_id)
    if definition.scope is ExecutionScope.SESSION:
        return (definition.executor_id, task.subject_id, task.session_id or "single")
    acquisition = task.acquisition_ids[0] if task.acquisition_ids else task.task_id
    return (definition.executor_id, task.subject_id, task.session_id or "single", acquisition)


def _execution_id(key: tuple[str, ...]) -> str:
    readable = "--".join(_safe_label(part) for part in key)
    return f"exec--{readable[:180]}--{_fingerprint(key)[:12]}"


def _format_values(
    layout: RunLayout,
    definition: ExecutorDefinition,
    task: ExecutableTask | None,
    context_path: str,
) -> dict[str, str]:
    subject_id = task.subject_id if task else None
    session_id = task.session_id if task else None
    plugins = set(task.plugin_ids if task else [])
    spatial = []
    for plugin_id, modality in (
        ("mri.t2w", "t2w"),
        ("mri.flair", "flair"),
        ("mri.pdw", "pd"),
        ("ct", "ct"),
    ):
        if plugin_id in plugins:
            spatial.append(modality)
    image = definition.image or ""
    image = image.format(container_root=layout.container_root)
    bids_inputs = list(task.metadata.get("bids_inputs", [])) if task else []
    bids_input = (
        str(Path("/source-data") / Path(str(bids_inputs[0])).relative_to(layout.bids_root))
        if len(bids_inputs) == 1
        else ""
    )
    primary = task.metadata.get("primary_anatomical_bids_input") if task else None
    primary_t1w = (
        str(Path("/source-data") / Path(str(primary)).relative_to(layout.bids_root))
        if primary
        else ""
    )
    return {
        "dataset_id": layout.dataset_id,
        "source_root": layout.source_root,
        "bids_root": layout.bids_root,
        "derivatives_root": layout.derivatives_root,
        "work_root": layout.work_root,
        "state_path": layout.state_path,
        "agentic_project": layout.agentic_project,
        "neuroprep_project": layout.neuroprep_project,
        "container_root": layout.container_root,
        "container_runtime": str(paths.container_runtime()),
        "agentic_python": sys.executable,
        "agentic_resources": str(paths.package_resources()),
        "subject": (subject_id or "dataset").removeprefix("sub-"),
        "session": (session_id or "single").removeprefix("ses-"),
        "context_path": context_path,
        "image": image,
        "freesurfer_license": layout.freesurfer_license or "__MISSING_FREESURFER_LICENSE__",
        "task_work_dir": task.work_dir if task else layout.work_root,
        "task_tmp": str(Path(task.work_dir if task else layout.work_root) / "tmp"),
        "task_var_tmp": str(Path(task.work_dir if task else layout.work_root) / "var-tmp"),
        "task_home": str(Path(task.work_dir if task else layout.work_root) / "home"),
        "cpus": str(task.cpus if task else 1),
        "memory_gb": str(task.memory_gb if task else 4),
        "spatial_modalities": " ".join(spatial),
        "primary_t1w": primary_t1w,
        "bids_input": bids_input,
        "acquisition": (
            _safe_label(task.acquisition_ids[0]) if task and task.acquisition_ids else ""
        ),
    }


def _format_argv(template: tuple[str, ...], values: dict[str, str]) -> list[str]:
    result: list[str] = []
    for token in template:
        if token == "{spatial_modalities}":
            result.extend(item for item in values["spatial_modalities"].split() if item)
        else:
            rendered = token.format(**values)
            if rendered:
                result.append(rendered)
    return result


def _source_bval_path(source_path: str) -> Path:
    path = Path(source_path)
    if path.name.endswith(".nii.gz"):
        return path.with_name(f"{path.name[:-7]}.bval")
    return path.with_suffix(".bval")


def _qsiprep_sampling_strategy(source_paths: list[str]) -> tuple[str, dict[str, Any]]:
    """Choose Eddy for shelled data and SHORELine for DSI-like q-space sampling.

    FSL Eddy supports single- and multi-shell acquisitions, but not DSI data.  A
    large number of nominal b-value levels is therefore a deterministic signal
    to use QSIPrep's 3dSHORE motion model instead.  Nominal levels are rounded
    to 100 s/mm2 so harmless scanner rounding does not create false shells.
    """

    evidence: list[dict[str, Any]] = []
    use_shoreline = False
    for source_path in source_paths:
        bval_path = _source_bval_path(source_path)
        if not bval_path.is_file():
            continue
        try:
            bvals = [float(item) for item in bval_path.read_text(encoding="utf-8").split()]
        except ValueError as exc:
            raise ValueError(f"invalid numeric b-values in {bval_path}") from exc
        nominal_levels = sorted(
            {
                int(round(value / 100.0) * 100)
                for value in bvals
                if value >= 50 and int(round(value / 100.0) * 100) > 0
            }
        )
        dsi_like = len(nominal_levels) > 6
        use_shoreline = use_shoreline or dsi_like
        evidence.append(
            {
                "source_bval": str(bval_path),
                "volume_count": len(bvals),
                "nominal_nonzero_b_values": nominal_levels,
                "dsi_like": dsi_like,
            }
        )
    return ("3dSHORE" if use_shoreline else "eddy"), {
        "selected_hmc_model": "3dSHORE" if use_shoreline else "eddy",
        "selection_rule": "3dSHORE when any series has more than six nominal nonzero b-values",
        "series_evidence": evidence,
    }


def _adapt_qsiprep_command(
    argv: list[str], source_paths: list[str]
) -> tuple[list[str], dict[str, Any]]:
    model, evidence = _qsiprep_sampling_strategy(source_paths)
    if model == "eddy":
        return argv, evidence
    adapted: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == "--eddy-config":
            index += 2
            continue
        adapted.append(argv[index])
        index += 1
    adapted.extend(("--hmc-model", "3dSHORE"))
    return adapted, evidence


def _task_scoped_container_argv(
    argv: list[str],
    definition: ExecutorDefinition,
    values: dict[str, str],
) -> list[str]:
    if definition.backend not in {ExecutorBackend.BIDS_APP, ExecutorBackend.CONTAINER}:
        return argv
    try:
        image_index = argv.index(values["image"])
    except ValueError as exc:
        raise ValueError(
            f"container executor {definition.executor_id} command omits its image"
        ) from exc
    isolation = [
        "--home",
        values["task_home"],
        "--bind",
        f"{values['task_tmp']}:/tmp",
        "--bind",
        f"{values['task_var_tmp']}:/var/tmp",
    ]
    return [*argv[:image_index], *isolation, *argv[image_index:]]


def _format_artifacts(
    requirements: tuple[ArtifactRequirement, ...], values: dict[str, str]
) -> list[ArtifactRequirement]:
    return [
        requirement.model_copy(
            update={"patterns": tuple(pattern.format(**values) for pattern in requirement.patterns)}
        )
        for requirement in requirements
    ]


def _dataset_control_task(
    definition: ExecutorDefinition,
    layout: RunLayout,
    *,
    depends_on: list[str],
    input_paths: list[str],
    metadata: dict[str, Any],
) -> ExecutableTask:
    task_id = _execution_id((definition.executor_id, "dataset"))
    context_path = str(Path(layout.work_root) / "contexts" / f"{task_id}.json")
    output_root = (
        layout.bids_root
        if definition.executor_id == "neuroprep-bids-init"
        else layout.derivatives_root
    )
    placeholder = ExecutableTask(
        task_id=task_id,
        logical_task_ids=[f"{layout.dataset_id}:{definition.stages[0]}"],
        executor_id=definition.executor_id,
        plugin_ids=list(definition.plugin_ids),
        stages=list(definition.stages),
        scope=ExecutionScope.DATASET,
        depends_on=sorted(depends_on),
        command=["placeholder"],
        input_paths=input_paths,
        output_root=output_root,
        work_dir=str(Path(layout.work_root) / "tasks" / task_id),
        resource_class="cpu",
        cpus=2,
        memory_gb=8,
        gpus=0,
        walltime_minutes=60,
        artifacts=[],
        fingerprint="pending",
        metadata={
            **metadata,
            "bids_root": layout.bids_root,
            "derivatives_root": layout.derivatives_root,
            "executor_version": definition.version,
            "executor_source": definition.source,
            "image_source": definition.image_source,
        },
    )
    values = _format_values(layout, definition, placeholder, context_path)
    payload = placeholder.model_dump(mode="json")
    payload["metadata"]["container_image"] = values["image"]
    payload["command"] = _format_argv(definition.command, values)
    payload["artifacts"] = [
        requirement.model_dump(mode="json")
        for requirement in _format_artifacts(definition.artifacts, values)
    ]
    payload["fingerprint"] = _fingerprint(payload)
    return ExecutableTask.model_validate(payload)


def _bind_consumers(
    grouped: dict[tuple[str, ...], list[TaskSpec]],
    definitions: dict[tuple[str, ...], ExecutorDefinition],
    warnings: list[str],
) -> dict[tuple[str, ...], tuple[str, ExecutorDefinition]]:
    """Give each consumer task the planned backend that provides its interface.

    No provider: an optional consumer is dropped with a warning; a required one is an error
    (selection only routes away from a provider when the alternative takes over the
    consumer's required stages, e.g. MRtrix3 fitting tensors itself instead of QSIRecon).
    """
    providers: dict[tuple[str | None, str], list[ExecutorDefinition]] = defaultdict(list)
    for key, definition in definitions.items():
        for interface in definition.provides:
            providers[(grouped[key][0].subject_id, interface)].append(definition)
    bound: dict[tuple[str, ...], tuple[str, ExecutorDefinition]] = {}
    for key in [key for key, definition in definitions.items() if definition.consumes]:
        definition, members = definitions[key], grouped[key]
        if len(definition.consumes) != 1:
            raise ValueError(f"{definition.executor_id} must consume exactly one interface")
        interface = definition.consumes[0]
        found = providers.get((members[0].subject_id, interface), [])
        if not found:
            if any(member.required for member in members):
                raise ValueError(
                    f"executor {definition.executor_id} for {members[0].subject_id} needs a "
                    f"planned backend that provides {interface}"
                )
            warnings.append(
                f"dropped optional {definition.executor_id} for {members[0].subject_id}: "
                f"no selected backend provides {interface}"
            )
            del grouped[key], definitions[key]
            continue
        if len(found) > 1:
            # shortcut: no catalog interface has two providers yet; when one does, run one
            # consumer per provider in the pilot so compatibility is qualified too.
            names = ", ".join(item.executor_id for item in found)
            raise ValueError(f"{interface} has several providers ({names}); not supported yet")
        bound[key] = (interface, found[0])
    return bound


def compile_execution_plan(
    graph: CanonicalAcquisitionGraph,
    logical_plan: PlanResult,
    layout: RunLayout,
    catalog: ExecutorCatalog,
    *,
    run_policy: dict[str, Any] | None = None,
    policy_digest: str | None = None,
    backend_decisions: list[dict[str, Any]] | None = None,
    backend_qualification_tests: list[dict[str, Any]] | None = None,
    planned_smoke_tests: list[dict[str, Any]] | None = None,
    backend_evidence: BackendEvidenceBinding | None = None,
    qualification_candidates: Collection[str] = (),
) -> ExecutionPlan:
    """Compile the executable DAG.

    Each (plugin, stage) runs the executor its backend decision selected (the catalog
    default otherwise). A comparison pilot passes ``qualification_candidates``: when two or
    more executors of one route are candidates, all of them run, as optional tasks, so a
    failing alternative loses the comparison instead of failing the pilot.
    """
    if graph.dataset.dataset_id != layout.dataset_id:
        raise ValueError("graph and run layout dataset ids differ")

    acquisitions = {item.acquisition_id: item for item in graph.acquisitions}
    executable: list[ExecutableTask] = []
    logical_to_execution: dict[str, str] = {}
    bids_execution: dict[str, str] = {}

    staged_acquisitions = [item for item in graph.acquisitions if item.bids is not None]
    bids_init_id: str | None = None
    if staged_acquisitions:
        init_definition = catalog.by_id("neuroprep-bids-init")
        init_task = _dataset_control_task(
            init_definition,
            layout,
            depends_on=[],
            input_paths=[],
            metadata={
                "dataset_id": graph.dataset.dataset_id,
                "bids_version": graph.dataset.bids_version,
            },
        )
        executable.append(init_task)
        bids_init_id = init_task.task_id

    bids_definition = catalog.resolve("core.bids", "bids_stage")
    for acquisition in staged_acquisitions:
        relative = bids_relative_path(acquisition)
        target = str(Path(layout.bids_root) / relative)
        key = (bids_definition.executor_id, acquisition.acquisition_id)
        task_id = _execution_id(key)
        context_path = str(Path(layout.work_root) / "contexts" / f"{task_id}.json")
        placeholder = ExecutableTask(
            task_id=task_id,
            logical_task_ids=[f"{acquisition.acquisition_id}:bids_stage"],
            executor_id=bids_definition.executor_id,
            plugin_ids=["core.bids"],
            stages=["bids_stage"],
            scope=ExecutionScope.ACQUISITION,
            subject_id=acquisition.subject_id,
            session_id=acquisition.session_id,
            acquisition_ids=[acquisition.acquisition_id],
            depends_on=[bids_init_id] if bids_init_id else [],
            command=["placeholder"],
            input_paths=[acquisition.source.source_path],
            output_root=layout.bids_root,
            work_dir=str(Path(layout.work_root) / "tasks" / task_id),
            resource_class="cpu",
            cpus=1,
            memory_gb=4,
            gpus=0,
            walltime_minutes=60,
            artifacts=[],
            fingerprint="pending",
            metadata={
                "source_path": acquisition.source.source_path,
                "bids_path": target,
                "bids_relative_path": str(relative),
                "bids_sidecar": acquisition.bids.sidecar,
                "source_metadata": acquisition.source.metadata,
                "source_format": acquisition.source.file_format,
                "selected_plugin_id": acquisition.selected_plugin_id,
                "bids_datatype": acquisition.bids.datatype,
                "bids_suffix": acquisition.bids.suffix,
                "support_level": acquisition.bids.support_level,
            },
        )
        values = _format_values(layout, bids_definition, placeholder, context_path)
        values["bids_path"] = target
        payload = placeholder.model_dump(mode="json")
        payload["command"] = _format_argv(bids_definition.command, values)
        payload["artifacts"] = [
            requirement.model_dump(mode="json")
            for requirement in _format_artifacts(bids_definition.artifacts, values)
        ]
        payload["fingerprint"] = _fingerprint(payload)
        task = ExecutableTask.model_validate(payload)
        executable.append(task)
        bids_execution[acquisition.acquisition_id] = task_id

    bids_validator_id: str | None = None
    if bids_execution:
        validator_definition = catalog.by_id("bids-validator")
        staged_paths = [
            str(task.metadata["bids_path"])
            for task in executable
            if task.executor_id == "neuroprep-bids-stage"
        ]
        validator_task = _dataset_control_task(
            validator_definition,
            layout,
            depends_on=sorted(bids_execution.values()),
            input_paths=staged_paths,
            metadata={
                "dataset_id": graph.dataset.dataset_id,
                "bids_version": graph.dataset.bids_version,
            },
        )
        executable.append(validator_task)
        bids_validator_id = validator_task.task_id

    selected = {
        str(item["selected_executor_id"])
        for item in (backend_decisions or [])
        if item.get("selected_executor_id")
    }
    grouped: dict[tuple[str, ...], list[TaskSpec]] = defaultdict(list)
    definitions: dict[tuple[str, ...], ExecutorDefinition] = {}
    candidate_keys: set[tuple[str, ...]] = set()
    committed_keys: set[tuple[str, ...]] = set()  # also does non-compared (required) work
    for task in logical_plan.tasks:
        routed = catalog.resolve(task.plugin_id, task.stage, selected)
        compared = [
            item
            for item in catalog.alternatives(task.plugin_id, task.stage)
            if item.executor_id in qualification_candidates
        ]
        chosen = list({item.executor_id: item for item in (routed, *compared)}.values())
        for definition in chosen:
            key = _scope_key(definition, task)
            grouped[key].append(task)
            definitions[key] = definition
            is_candidate = len(chosen) > 1 and definition in compared
            (candidate_keys if is_candidate else committed_keys).add(key)
    candidate_keys -= committed_keys
    warnings: list[str] = []
    consumed_from = _bind_consumers(grouped, definitions, warnings)

    pending_dependency_ids: dict[str, set[str]] = {}
    for key, members in grouped.items():
        definition = definitions[key]
        task_id = _execution_id(key)
        for member in members:
            # Pilot alternatives share a logical task; dependents follow the catalog default.
            logical_to_execution.setdefault(member.task_id, task_id)
        pending_dependency_ids[task_id] = {
            dependency for member in members for dependency in member.depends_on
        }

        acquisition_ids = sorted(
            {acquisition_id for member in members for acquisition_id in member.acquisition_ids}
        )
        source_paths = [
            acquisitions[item].source.source_path
            for item in acquisition_ids
            if item in acquisitions
        ]
        bids_inputs = [
            str(Path(layout.bids_root) / bids_relative_path(acquisitions[item]))
            for item in acquisition_ids
            if item in acquisitions and acquisitions[item].bids is not None
        ]
        source_metadata = {
            item: acquisitions[item].source.metadata
            for item in acquisition_ids
            if item in acquisitions
        }
        resource = _maximum_resource([member.resource for member in members])
        required = any(member.required for member in members) and key not in candidate_keys
        subject_id = members[0].subject_id
        sessions = {member.session_id for member in members}
        session_id = sessions.pop() if len(sessions) == 1 else None
        context_path = str(Path(layout.work_root) / "contexts" / f"{task_id}.json")
        placeholder = ExecutableTask(
            task_id=task_id,
            logical_task_ids=sorted(member.task_id for member in members),
            executor_id=definition.executor_id,
            plugin_ids=sorted({member.plugin_id for member in members}),
            stages=sorted({member.stage for member in members}),
            scope=definition.scope,
            subject_id=subject_id,
            session_id=session_id,
            acquisition_ids=acquisition_ids,
            command=["placeholder"],
            input_paths=source_paths,
            output_root=layout.derivatives_root,
            work_dir=str(Path(layout.work_root) / "tasks" / task_id),
            resource_class=resource.resource_class,
            cpus=resource.cpus,
            memory_gb=resource.memory_gb,
            gpus=resource.gpus,
            walltime_minutes=resource.walltime_minutes,
            required=required,
            artifacts=[],
            fingerprint="pending",
            metadata={
                "source_metadata": source_metadata,
                "bids_inputs": bids_inputs,
                "bids_root": layout.bids_root,
                "derivatives_root": layout.derivatives_root,
                "executor_version": definition.version,
                "executor_source": definition.source,
                "required_metadata": list(definition.required_metadata),
                "image_source": definition.image_source,
                "notes": definition.notes,
                **({"qualification_candidate": True} if key in candidate_keys else {}),
            },
        )
        if definition.executor_id in {"smriprep", "deepprep", "fastsurfer", "synthseg"}:
            primary = select_primary_t1(
                [
                    (item, acquisitions[item].source.metadata)
                    for item in acquisition_ids
                    if item in acquisitions
                ]
            )
            primary_acquisition = acquisitions[primary.acquisition_id]
            placeholder.metadata.update(
                {
                    "primary_anatomical_acquisition_id": primary.acquisition_id,
                    "primary_anatomical_bids_input": str(
                        Path(layout.bids_root) / bids_relative_path(primary_acquisition)
                    ),
                    "primary_anatomical_score": primary.score,
                    "primary_anatomical_reasons": list(primary.reasons),
                    "primary_anatomical_candidate_count": len(acquisition_ids),
                }
            )
        values = _format_values(layout, definition, placeholder, context_path)
        if key in consumed_from:
            interface, provider = consumed_from[key]
            values[interface] = provider.provides[interface].format(**values)
            placeholder.metadata["consumes_from"] = provider.executor_id
        payload = placeholder.model_dump(mode="json")
        rendered_command = _format_argv(definition.command, values)
        if definition.executor_id == "qsiprep":
            rendered_command, dwi_sampling = _adapt_qsiprep_command(
                rendered_command, source_paths
            )
            payload["metadata"]["dwi_sampling"] = dwi_sampling
        payload["command"] = _task_scoped_container_argv(
            rendered_command, definition, values
        )
        payload["environment"] = {
            key: value.format(**values) for key, value in definition.environment.items()
        }
        payload["artifacts"] = [
            requirement.model_dump(mode="json")
            for requirement in _format_artifacts(definition.artifacts, values)
        ]
        payload["fingerprint"] = _fingerprint(payload)
        executable.append(ExecutableTask.model_validate(payload))

    by_id = {task.task_id: task for task in executable}
    for task_id, logical_dependencies in pending_dependency_ids.items():
        task = by_id[task_id]
        dependencies = {
            logical_to_execution[dependency]
            for dependency in logical_dependencies
            if dependency in logical_to_execution
        }
        if bids_validator_id:
            dependencies.add(bids_validator_id)
        dependencies.update(
            bids_execution[item] for item in task.acquisition_ids if item in bids_execution
        )
        dependencies.discard(task_id)
        by_id[task_id] = task.model_copy(update={"depends_on": sorted(dependencies)})

    by_subject_executor: dict[tuple[str, str], list[str]] = defaultdict(list)
    for task in by_id.values():
        if task.subject_id is not None:
            by_subject_executor[(task.subject_id, task.executor_id)].append(task.task_id)

    for task_id, task in list(by_id.items()):
        definition = catalog.by_id(task.executor_id)
        dependencies = set(task.depends_on)
        provider_id = task.metadata.get("consumes_from")
        if provider_id:
            dependencies.update(by_subject_executor.get((task.subject_id, provider_id), []))
        for required_executor_id in definition.requires_executor_ids:
            if task.subject_id is None:
                raise ValueError(
                    f"executor {task.executor_id} requires {required_executor_id} "
                    "but has no subject scope"
                )
            candidates = by_subject_executor.get((task.subject_id, required_executor_id), [])
            if not candidates:
                raise ValueError(
                    f"executor {task.executor_id} for {task.subject_id} requires "
                    f"a planned {required_executor_id} task"
                )
            dependencies.update(candidates)
        dependencies.discard(task_id)
        by_id[task_id] = task.model_copy(update={"depends_on": sorted(dependencies)})

    for task_id, task in list(by_id.items()):
        by_id[task_id] = task.model_copy(update={"fingerprint": executable_task_fingerprint(task)})
    ordered = _topological_tasks(list(by_id.values()))
    required_ids = sorted({task.executor_id for task in ordered if task.required})
    optional_ids = sorted({task.executor_id for task in ordered if not task.required})
    used_executor_ids = set(required_ids) | set(optional_ids)
    relevant_decisions = [
        item
        for item in (backend_decisions or [])
        if item.get("selected_executor_id") in used_executor_ids
    ]
    compared_capabilities = {item.get("capability") for item in relevant_decisions}
    relevant_smoke_tests = [
        item
        for item in (planned_smoke_tests or [])
        if item.get("executor_id") in used_executor_ids
        or item.get("capability") in compared_capabilities
    ]
    return ExecutionPlan(
        dataset_id=layout.dataset_id,
        generated_at=datetime.now(UTC).isoformat(),
        catalog_digest=catalog.digest(),
        layout=layout,
        tasks=ordered,
        required_executor_ids=required_ids,
        optional_executor_ids=optional_ids,
        run_policy=run_policy or {},
        policy_digest=policy_digest,
        backend_decisions=relevant_decisions,
        backend_qualification_tests=backend_qualification_tests or [],
        planned_smoke_tests=relevant_smoke_tests,
        backend_evidence=backend_evidence,
        warnings=warnings,
    )


def _maximum_resource(resources: list[ResourceRequest]) -> ResourceRequest:
    priority = {"cpu": 0, "high_memory": 1, "gpu": 2}
    selected_class = max((item.resource_class for item in resources), key=priority.__getitem__)
    return ResourceRequest(
        resource_class=selected_class,  # type: ignore[arg-type]
        cpus=max(item.cpus for item in resources),
        memory_gb=max(item.memory_gb for item in resources),
        gpus=max(item.gpus for item in resources),
        walltime_minutes=max(item.walltime_minutes for item in resources),
    )


def _topological_tasks(tasks: list[ExecutableTask]) -> list[ExecutableTask]:
    remaining = {task.task_id: task for task in tasks}
    ordered: list[ExecutableTask] = []
    emitted: set[str] = set()
    while remaining:
        ready = sorted(
            task_id for task_id, task in remaining.items() if set(task.depends_on) <= emitted
        )
        if not ready:
            unresolved = ", ".join(sorted(remaining)[:8])
            raise ValueError(f"execution graph has a cycle or missing dependency: {unresolved}")
        for task_id in ready:
            task = remaining.pop(task_id)
            ordered.append(task)
            emitted.add(task_id)
    return ordered


def save_execution_plan(plan: ExecutionPlan, destination: str) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    contexts = Path(plan.layout.work_root) / "contexts"
    contexts.mkdir(parents=True, exist_ok=True)
    for task in plan.tasks:
        context = contexts / f"{task.task_id}.json"
        temporary = context.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(task.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(context)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
