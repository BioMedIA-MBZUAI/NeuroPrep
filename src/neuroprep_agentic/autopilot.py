"""Prompt-and-path autopilot: one request in, a qualified and released run out.

``run_autopilot`` chains NeuroPrep's existing tools without a human in the loop:

1. **policy**   – the language model compiles the request into one typed ``RunPolicy``
                  (reused by every later stage, so the model is sampled exactly once);
2. **probe**    – deterministic inventory, classification and plan compilation, which
                  lists the backend smoke tests the target run requires;
3. **pilot**    – the smallest set of participants that exercises every required backend
                  is cut from the inventory and executed end to end on this host;
4. **evidence** – the terminal pilot is audited and signed (HMAC) against the target plan;
5. **run**      – the production plan is compiled with the verified evidence, initialized,
                  executed (locally or through Slurm) and released.

Autonomy replaces the human approval step, not the checks: every gate that guards a manual
run (schema validation, catalog/executor/image identity, terminal pilot audit, clean git
provenance, preflight, artifact and QC contracts, release) still decides. The report records
``approval: autonomous`` so an autonomous run is never mistaken for an operator-signed one.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .backend_evidence import (
    build_backend_evidence,
    create_signing_key,
    read_signing_key,
    write_evidence_document,
)
from .execution_models import ExecutionPlan
from .policy import HardwareProfile, compile_prompt_policy, detect_hardware_profile
from .prompt_run import prepare_prompt_run, resolve_prompt_paths
from .release import finalize_execution_run
from .runtime import run_worker
from .services import ManifestDocument, load_manifest
from .settings import validated_agentic_project, validated_cluster_path
from .state import RunState

ExecuteMode = Literal["local", "slurm", "plan-only"]
TERMINAL_FAILURES = frozenset({"failed", "blocked"})


class AutopilotError(RuntimeError):
    """A stage could not complete; ``report`` says which and why."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def qualification_root(output_root: Path) -> Path:
    """Sibling of the run root, so the run root itself stays empty until production."""

    return output_root.parent / f"{output_root.name}.neuroprep-qualification"


def select_pilot_subjects(
    plan: dict[str, Any],
    required_executors: set[str],
    *,
    max_subjects: int = 3,
) -> tuple[list[str], set[str]]:
    """Greedy minimum set of subjects whose tasks exercise every required executor.

    Ties prefer the subject with fewer tasks (cheaper pilot), then the lower id, so the
    choice is deterministic for a given plan. Returns the subjects and any executors no
    subject could cover within ``max_subjects``.
    """

    coverage: dict[str, set[str]] = {}
    cost: dict[str, int] = {}
    for task in plan.get("tasks", []):
        subject = task.get("subject_id")
        if not subject:
            continue
        coverage.setdefault(subject, set()).add(str(task.get("executor_id")))
        cost[subject] = cost.get(subject, 0) + 1
    remaining = set(required_executors)
    chosen: list[str] = []
    while remaining and len(chosen) < max_subjects:
        candidates = [
            subject
            for subject in coverage
            if subject not in chosen and coverage[subject] & remaining
        ]
        if not candidates:
            break
        best = min(
            candidates,
            key=lambda subject: (-len(coverage[subject] & remaining), cost[subject], subject),
        )
        chosen.append(best)
        remaining -= coverage[best]
    return chosen, remaining


def pilot_manifest(
    manifest: ManifestDocument, graph: dict[str, Any], subject_ids: list[str]
) -> ManifestDocument:
    """Restrict the inventory to the source records of the chosen (pseudonymous) subjects."""

    wanted = set(subject_ids)
    paths = {
        acquisition["source"]["source_path"]
        for subject in graph.get("subjects", [])
        if subject.get("subject_id") in wanted
        for session in subject.get("sessions", [])
        for acquisition in session.get("acquisitions", [])
    }
    records = [record for record in manifest.records if record.source_path in paths]
    if not records:
        raise ValueError("pilot subjects map to no source records")
    return manifest.model_copy(update={"records": records})


def _execute_locally(
    plan_path: Path, *, worker: Callable[..., dict[str, int]], idle_seconds: int
) -> dict[str, Any]:
    plan = ExecutionPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    classes = {task.resource_class for task in plan.tasks}
    counts = worker(
        plan.layout.state_path,
        f"autopilot-{os.getpid()}",
        classes,
        idle_seconds=idle_seconds,
        exit_when_idle=True,
    )
    summary = RunState(plan.layout.state_path).summary()
    return {"worker": counts, "state": summary}


def _failed_executors(summary: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(row["executor_id"])
            for row in summary.get("active", [])
            if str(row.get("status")) in TERMINAL_FAILURES
        }
    )


def run_autopilot(
    request: str,
    *,
    source_root: str | None = None,
    output_root: str | None = None,
    agentic_project: str | None = None,
    hardware_profile: HardwareProfile | None = None,
    signing_key: str | None = None,
    freesurfer_license: str | None = None,
    execute: ExecuteMode = "local",
    gpu_workers: int = 2,
    max_pilot_subjects: int = 3,
    allow_quarantine: bool = False,
    allow_unsupported: bool = False,
    idle_seconds: int = 10,
    worker: Callable[..., dict[str, int]] = run_worker,
) -> dict[str, Any]:
    """Run the whole prompt → qualification → execution → release chain unattended."""

    binding = resolve_prompt_paths(request, source_root=source_root, output_root=output_root)
    source = Path(binding.source_root)
    output = Path(binding.output_root)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"run root already exists and is not empty: {output}")
    qualification = qualification_root(output)
    validated_cluster_path(str(qualification))
    if qualification.exists() and any(qualification.iterdir()):
        raise ValueError(f"qualification root already exists and is not empty: {qualification}")
    qualification.mkdir(parents=True, exist_ok=True)
    project = str(validated_agentic_project(agentic_project))

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "approval": "autonomous",
        "started_at": _now(),
        "request": request.strip(),
        "source_root": str(source),
        "run_root": str(output),
        "qualification_root": str(qualification),
        "execute": execute,
        "stages": {},
    }
    report_path = qualification / "autopilot-report.json"

    def stage(name: str, **values: Any) -> None:
        report["stages"][name] = {"finished_at": _now(), **values}
        _write_json(report_path, report)

    def fail(name: str, message: str, **values: Any) -> AutopilotError:
        stage(name, status="failed", error=message, **values)
        report["outcome"] = "failed"
        report["failed_stage"] = name
        _write_json(report_path, report)
        return AutopilotError(f"{name}: {message}", report)

    common = {
        "source_root": str(source),
        "agentic_project": project,
        "freesurfer_license": freesurfer_license,
        "allow_quarantine": allow_quarantine,
        "allow_unsupported": allow_unsupported,
    }

    # 1. one model decision for the whole run
    profile = hardware_profile or detect_hardware_profile()
    try:
        policy = compile_prompt_policy(request, profile)
    except Exception as error:  # noqa: BLE001 - surfaced verbatim in the report
        raise fail("policy", str(error)) from error
    stage("policy", status="passed", policy_digest=policy.digest(), hardware=profile.hostname)

    # 2. probe: what would the target run need?
    probe_root = qualification / "probe"
    try:
        probe = prepare_prompt_run(
            request,
            output_root=str(probe_root),
            hardware_profile=profile,
            policy=policy,
            **common,
        )
    except Exception as error:  # noqa: BLE001
        raise fail("probe", str(error)) from error
    target_plan_path = Path(probe["execution_plan"])
    target_plan = json.loads(target_plan_path.read_text(encoding="utf-8"))
    smoke_tests = list(target_plan.get("backend_qualification_tests") or [])
    required = {str(item["executor_id"]) for item in smoke_tests}
    stage(
        "probe",
        status="passed",
        task_count=probe["task_count"],
        record_count=probe["record_count"],
        required_smoke_tests=sorted(str(item["test_id"]) for item in smoke_tests),
    )

    evidence_path: Path | None = None
    key_path: str | None = signing_key
    if smoke_tests:
        # 3. pilot: the smallest representative cohort that exercises every backend
        subjects, uncovered = select_pilot_subjects(
            target_plan, required, max_subjects=max_pilot_subjects
        )
        if uncovered:
            raise fail(
                "pilot_selection",
                "no pilot within the subject budget exercises: " + ", ".join(sorted(uncovered)),
                subjects=subjects,
            )
        graph = json.loads((probe_root / "acquisition-graph.json").read_text(encoding="utf-8"))
        manifest = load_manifest(str(probe_root / "manifest.json"))
        subset = pilot_manifest(manifest, graph, subjects)
        subset_path = qualification / "pilot-manifest.json"
        _write_json(subset_path, subset.model_dump(mode="json"))
        stage("pilot_selection", status="passed", subjects=subjects, records=len(subset.records))

        pilot_root = qualification / "pilot"
        try:
            pilot = prepare_prompt_run(
                request,
                output_root=str(pilot_root),
                manifest_path=str(subset_path),
                hardware_profile=profile,
                policy=policy,
                **common,
            )
        except Exception as error:  # noqa: BLE001
            raise fail("pilot_plan", str(error)) from error
        if not pilot["ready"]:
            raise fail("pilot_plan", "pilot preflight failed", preflight=pilot["preflight"])
        pilot_plan_path = Path(pilot["execution_plan"])
        pilot_plan = ExecutionPlan.model_validate_json(pilot_plan_path.read_text())
        # The pilot *is* the qualification run, so it is initialized without prior evidence;
        # its outcome is what the evidence document will attest.
        RunState(pilot_plan.layout.state_path).initialize(pilot_plan, str(pilot_plan_path))
        executed = _execute_locally(pilot_plan_path, worker=worker, idle_seconds=idle_seconds)
        failed = _failed_executors(executed["state"])
        counts = executed["state"]["counts"]
        if failed or set(counts) - {"complete", "skipped"}:
            raise fail(
                "pilot_run",
                "pilot did not complete; failing backends: " + (", ".join(failed) or "none"),
                counts=counts,
                failed_executors=failed,
            )
        release_root = Path(pilot_plan.layout.derivatives_root) / "release"
        finalize_execution_run(
            pilot_plan, RunState(pilot_plan.layout.state_path), str(release_root)
        )
        stage("pilot_run", status="passed", counts=counts)

        # 4. evidence: audited and signed against the target plan
        if key_path is None:
            keys = qualification / "keys"
            keys.mkdir(parents=True, exist_ok=True)
            key_path = str(create_signing_key(keys / "autopilot.key"))
        key_id = f"autopilot-{target_plan['dataset_id']}-{_now().replace(':', '')}"
        try:
            document = build_backend_evidence(
                pilot_plan_path=pilot_plan_path,
                target_plan_path=target_plan_path,
                pilot_state_path=pilot_plan.layout.state_path,
                pilot_release_path=release_root / "release-summary.json",
                git_repository=project,
                signing_key=read_signing_key(key_path),
                key_id=key_id,
            )
        except Exception as error:  # noqa: BLE001
            hint = (
                " (commit or stash the NeuroPrep checkout; evidence binds a clean git commit)"
                if "clean git worktree" in str(error)
                else ""
            )
            raise fail("evidence", str(error) + hint) from error
        evidence_path = write_evidence_document(document, qualification / "evidence.json")
        stage(
            "evidence",
            status="passed",
            key_id=key_id,
            tests={item.test_id: item.status for item in document.tests},
            target_git_commit=document.target_git_commit,
        )
    else:
        stage("pilot_selection", status="skipped", reason="no backend smoke tests required")

    # 5. production run with the verified evidence
    try:
        result = prepare_prompt_run(
            request,
            output_root=str(output),
            manifest_path=str(probe_root / "manifest.json"),
            hardware_profile=profile,
            policy=policy,
            backend_evidence=str(evidence_path) if evidence_path else None,
            backend_evidence_key=key_path if evidence_path else None,
            initialize=execute != "plan-only",
            submit=execute == "slurm",
            gpu_workers=gpu_workers,
            **common,
        )
    except Exception as error:  # noqa: BLE001
        raise fail("run_plan", str(error)) from error
    stage("run_plan", status="passed", task_count=result["task_count"], ready=result["ready"])

    if execute == "local":
        plan_path = Path(result["execution_plan"])
        executed = _execute_locally(plan_path, worker=worker, idle_seconds=idle_seconds)
        counts = executed["state"]["counts"]
        failed = _failed_executors(executed["state"])
        if failed or set(counts) - {"complete", "skipped"}:
            raise fail("run", "run did not complete", counts=counts, failed_executors=failed)
        plan = ExecutionPlan.model_validate_json(plan_path.read_text())
        release = finalize_execution_run(
            plan,
            RunState(plan.layout.state_path),
            str(Path(plan.layout.derivatives_root) / "release"),
        )
        stage("run", status="passed", counts=counts)
        stage("release", status="passed", summary=release)
        report["outcome"] = "released"
    elif execute == "slurm":
        stage("run", status="submitted", launch=result["launch"])
        report["outcome"] = "submitted"
    else:
        report["outcome"] = "planned"
    report["finished_at"] = _now()
    report["execution_plan"] = result["execution_plan"]
    _write_json(report_path, report)
    _write_json(output / "autopilot-report.json", report)
    return report
