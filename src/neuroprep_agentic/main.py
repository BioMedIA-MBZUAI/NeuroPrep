from __future__ import annotations

import argparse
import json
import os

from .backend_evidence import (
    BackendEvidenceDocument,
    build_backend_evidence,
    create_signing_key,
    current_git_commit,
    read_signing_key,
    revalidate_execution_evidence,
    validate_backend_evidence,
    write_evidence_document,
)
from .compiler import compile_execution_plan, save_execution_plan
from .control import run_control_flow, validate_control_artifact
from .execution_models import ExecutionPlan, RunLayout
from .executors import builtin_executor_catalog
from .images import (
    catalog_image_inventory,
    image_inventory,
    prefetch_catalog_images,
    prefetch_images,
)
from .openjev import (
    OpenJevClient,
    check_boolean,
    choose_next_action,
    route_acquisition,
    score_urgency,
)
from .policy import (
    RunPolicy,
    compile_prompt_policy,
    detect_hardware_profile,
    load_hardware_profile,
    load_policy,
    plan_backends,
)
from .preflight import preflight_execution_plan
from .prompt_run import PromptPathBinding, prepare_prompt_run
from .qualification import compile_backend_evidence
from .registry import builtin_registry
from .release import finalize_execution_run
from .runtime import run_worker
from .runtime_assets import prepare_runtime_assets, runtime_asset_inventory
from .runtime_check import runtime_check
from .services import build_graph_and_plan, build_ingest_plan, load_manifest
from .settings import (
    DEFAULT_AGENTIC_PROJECT,
    validated_agentic_project,
    validated_cluster_path,
)
from .state import RunState


def _load_execution_plan(path: str) -> ExecutionPlan:
    candidate = validated_cluster_path(path, must_exist=True)
    return ExecutionPlan.model_validate_json(candidate.read_text(encoding="utf-8"))


def _write_or_print(payload: object, output: str | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    if output is None:
        print(rendered, end="")
        return
    destination = validated_cluster_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    print(destination)


def _child_parent_pair(value: str) -> tuple[str, str]:
    child, separator, parent = value.partition("=")
    if separator != "=" or not child.strip() or not parent.strip():
        raise argparse.ArgumentTypeError("expected CHILD_TASK_ID=PARENT_TASK_ID")
    return child.strip(), parent.strip()


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Dataset-agnostic multimodal NeuroPrep control plane"
    )
    from . import __version__

    parser.add_argument("--version", action="version", version=f"neuroprep-agentic {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    coverage = subparsers.add_parser("coverage", help="Show dataset and modality plugins")
    coverage.add_argument("--output")

    ingest = subparsers.add_parser("ingest-plan", help="Classify a manifest and build its DAG")
    ingest.add_argument("manifest")
    ingest.add_argument("--identity-key", default=os.getenv("NEUROPREP_IDENTITY_KEY"))
    ingest.add_argument("--output")

    crew = subparsers.add_parser("crew", help="Run the Ollama-backed CrewAI Flow")
    crew.add_argument(
        "--workflow",
        choices=("coverage", "ingest_plan", "execution_review"),
        default="coverage",
    )
    crew.add_argument("--request", default="Review coverage without inventing support.")

    jev = subparsers.add_parser(
        "jev-decision", help="Run one bounded local OpenJev routing or recovery decision"
    )
    jev.add_argument("kind", choices=("route", "boolean", "urgency", "next_action"))
    jev.add_argument("state", help="JSON file containing a de-identified acquisition/task state")
    jev.add_argument("--endpoint")
    jev.add_argument("--model", default="openjev")
    jev.add_argument("--instructions", default="Is this state safe to continue automatically?")

    subparsers.add_parser(
        "runtime-check",
        help="Check Ollama, OpenJev, scientific tools, containers, and runtime assets",
    )

    interface = subparsers.add_parser(
        "interface", help="Serve the local browser interface for safe study planning"
    )
    interface.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
    interface.add_argument("--port", type=int, default=8787)
    interface.add_argument(
        "--agentic-project",
        help="Exact NeuroPrep-agentic git root recorded in compiled plans",
    )

    hardware = subparsers.add_parser(
        "hardware-profile", help="Record CPU, memory, GPU, CUDA, and container-runtime evidence"
    )
    hardware.add_argument("--output")

    prompt_policy = subparsers.add_parser(
        "prompt-policy", help="Compile a natural-language request into a constrained run policy"
    )
    request_group = prompt_policy.add_mutually_exclusive_group(required=True)
    request_group.add_argument("--request")
    request_group.add_argument("--request-file")
    prompt_policy.add_argument("--hardware-profile")
    prompt_policy.add_argument("--output", required=True)

    auto = subparsers.add_parser(
        "auto",
        help=(
            "Fully autonomous run: give a request naming the input and output paths; "
            "NeuroPrep plans, qualifies backends on a pilot, signs evidence, runs and releases"
        ),
    )
    auto_request = auto.add_mutually_exclusive_group(required=True)
    auto_request.add_argument("request", nargs="?", help="e.g. 'My data is at X; save to Y'")
    auto_request.add_argument("--request-file")
    auto.add_argument("--source-root", help="Override the input path found in the request")
    auto.add_argument("--output-root", help="Override the output path found in the request")
    auto.add_argument(
        "--execute",
        choices=("local", "slurm", "plan-only"),
        default="local",
        help="local: run on this host (e.g. inside a GPU job); slurm: submit workers",
    )
    auto.add_argument("--gpu-workers", type=int, default=2)
    auto.add_argument("--max-pilot-subjects", type=int, default=3)
    auto.add_argument("--agentic-project")
    auto.add_argument("--hardware-profile")
    auto.add_argument(
        "--signing-key",
        default=os.getenv("NEUROPREP_BACKEND_EVIDENCE_KEY"),
        help="Existing evidence key; a run-local 0600 key is created when omitted",
    )
    auto.add_argument(
        "--freesurfer-license",
        default=os.getenv("FS_LICENSE") or os.getenv("FREESURFER_LICENSE"),
    )
    auto.add_argument("--allow-quarantine", action="store_true")
    auto.add_argument("--allow-unsupported", action="store_true")

    prompt_run = subparsers.add_parser(
        "prompt-run",
        help="Bind prompt paths, inventory a dataset, compile a run, and optionally submit it",
    )
    prompt_run_request = prompt_run.add_mutually_exclusive_group(required=True)
    prompt_run_request.add_argument("--request")
    prompt_run_request.add_argument("--request-file")
    prompt_run.add_argument("--source-root")
    prompt_run.add_argument("--output-root")
    prompt_run.add_argument("--manifest")
    prompt_run.add_argument("--dataset-id")
    prompt_run.add_argument("--adapter-id")
    prompt_run.add_argument("--identity-key", default=os.getenv("NEUROPREP_IDENTITY_KEY"))
    prompt_run.add_argument(
        "--agentic-project",
        help=(
            "Exact NeuroPrep-agentic git root; CLI overrides "
            "NEUROPREP_AGENTIC_PROJECT and the production default"
        ),
    )
    prompt_run.add_argument("--hardware-profile")
    prompt_run.add_argument("--backend-evidence")
    prompt_run.add_argument(
        "--backend-evidence-key",
        default=os.getenv("NEUROPREP_BACKEND_EVIDENCE_KEY"),
    )
    prompt_run.add_argument(
        "--freesurfer-license",
        default=os.getenv("FS_LICENSE") or os.getenv("FREESURFER_LICENSE"),
    )
    prompt_run.add_argument("--execute-optional", action="store_true")
    prompt_run.add_argument("--allow-quarantine", action="store_true")
    prompt_run.add_argument("--allow-unsupported", action="store_true")
    prompt_run.add_argument("--initialize", action="store_true")
    prompt_run.add_argument("--submit", action="store_true")
    prompt_run.add_argument(
        "--openjev-review",
        action="store_true",
        help="Review unresolved acquisitions with local OpenJev before fail-closed validation",
    )
    prompt_run.add_argument(
        "--gpu-workers",
        type=int,
        default=2,
        help="GPU worker count passed to the Slurm launcher when --submit is used",
    )

    backend_plan = subparsers.add_parser(
        "backend-plan", help="Select integrated backends and plan bounded qualification tests"
    )
    backend_plan.add_argument("--policy", required=True)
    backend_plan.add_argument("--hardware-profile")
    backend_plan.add_argument("--evidence")
    backend_plan.add_argument("--output")

    qualification = subparsers.add_parser(
        "qualification-evidence",
        help="Compile terminal representative-run evidence for adaptive backend selection",
    )
    qualification.add_argument("plan")
    qualification.add_argument("--output", required=True)

    evidence_builder = subparsers.add_parser(
        "build-backend-evidence",
        help="Build signed, target-bound evidence from a terminal representative run",
    )
    evidence_builder.add_argument("--pilot-plan", required=True)
    evidence_builder.add_argument("--pilot-state", required=True)
    evidence_builder.add_argument("--pilot-release", required=True)
    evidence_builder.add_argument("--target-plan", required=True)
    evidence_builder.add_argument(
        "--git-repository",
        default=os.getenv("NEUROPREP_AGENTIC_PROJECT", DEFAULT_AGENTIC_PROJECT),
    )
    evidence_builder.add_argument("--signing-key", required=True)
    evidence_builder.add_argument("--key-id", required=True)
    evidence_builder.add_argument("--output", required=True)

    evidence_key = subparsers.add_parser(
        "create-backend-evidence-key",
        help="Create a mode-0600 evidence signing key on approved cluster storage",
    )
    evidence_key.add_argument("--output", required=True)

    schema = subparsers.add_parser("schemas", help="Export core JSON Schemas")
    schema.add_argument("--output", required=True)

    validate = subparsers.add_parser(
        "validate-control-artifact",
        help="Recompute and verify a deterministic CrewAI control artifact",
    )
    validate.add_argument("artifact")
    validate.add_argument(
        "--workflow",
        choices=("coverage", "ingest_plan", "execution_review"),
        required=True,
    )

    executors = subparsers.add_parser(
        "executor-coverage", help="Audit deterministic executor coverage"
    )
    executors.add_argument("--output")

    compile_run = subparsers.add_parser(
        "compile-run", help="Compile a manifest into a fingerprinted executable DAG"
    )
    compile_run.add_argument("manifest")
    compile_run.add_argument("--identity-key", default=os.getenv("NEUROPREP_IDENTITY_KEY"))
    compile_run.add_argument("--source-root", required=True)
    compile_run.add_argument("--bids-root", required=True)
    compile_run.add_argument("--derivatives-root", required=True)
    compile_run.add_argument("--work-root", required=True)
    compile_run.add_argument("--state-path", required=True)
    compile_run.add_argument(
        "--agentic-project",
        help=(
            "Exact NeuroPrep-agentic git root; CLI overrides "
            "NEUROPREP_AGENTIC_PROJECT and the production default"
        ),
    )
    compile_run.add_argument("--freesurfer-license")
    compile_run.add_argument("--execute-optional", action="store_true")
    compile_run.add_argument("--policy")
    compile_run.add_argument("--hardware-profile")
    compile_run.add_argument("--backend-evidence")
    compile_run.add_argument(
        "--backend-evidence-key",
        default=os.getenv("NEUROPREP_BACKEND_EVIDENCE_KEY"),
    )
    compile_run.add_argument("--allow-quarantine", action="store_true")
    compile_run.add_argument("--allow-unsupported", action="store_true")
    compile_run.add_argument("--output", required=True)

    preflight = subparsers.add_parser("preflight-run", help="Fail-closed run preflight")
    preflight.add_argument("plan")
    preflight.add_argument("--skip-input-check", action="store_true")
    preflight.add_argument("--output")

    images = subparsers.add_parser(
        "prefetch-images", help="Inventory or atomically pull plan images"
    )
    images.add_argument("plan")
    images.add_argument("--execute", action="store_true")
    images.add_argument("--output")

    catalog_images = subparsers.add_parser(
        "catalog-images", help="Inventory, validate, or pull every catalog image"
    )
    catalog_images.add_argument("--execute", action="store_true")
    catalog_images.add_argument("--inspect", action="store_true")
    catalog_images.add_argument("--output")

    assets = subparsers.add_parser(
        "prepare-assets", help="Inventory or atomically stage pinned runtime assets"
    )
    assets.add_argument("plan")
    assets.add_argument("--execute", action="store_true")
    assets.add_argument("--output")
    initialize = subparsers.add_parser("run-init", help="Initialize the durable run database")
    initialize.add_argument("plan")
    initialize.add_argument(
        "--backend-evidence-key",
        default=os.getenv("NEUROPREP_BACKEND_EVIDENCE_KEY"),
    )
    initialize.add_argument("--allow-incomplete-preflight", action="store_true")

    worker = subparsers.add_parser("worker", help="Run a leased deterministic worker")
    worker.add_argument("--state", required=True)
    worker.add_argument("--worker-id", required=True)
    worker.add_argument(
        "--resource-class",
        action="append",
        choices=("cpu", "gpu", "high_memory"),
        required=True,
    )
    worker.add_argument("--lease-seconds", type=int, default=3600)
    worker.add_argument("--max-tasks", type=int)
    worker.add_argument("--stay-alive", action="store_true")

    recover = subparsers.add_parser(
        "recover-stale-task",
        help="Dry-run or compare-and-swap one exact expired task back to pending",
    )
    recover.add_argument("--state", required=True)
    recover.add_argument("--task-id", required=True)
    recover.add_argument("--expected-worker-id", required=True)
    recover.add_argument("--expected-attempt", type=int, required=True)
    recover.add_argument("--expected-lease-expires-at", required=True)
    recover.add_argument("--finished-at", required=True)
    recover.add_argument("--reason", required=True)
    recover.add_argument("--log-path", required=True)
    recover.add_argument("--return-code", type=int)
    recover_mode = recover.add_mutually_exclusive_group(required=True)
    recover_mode.add_argument("--dry-run", action="store_true")
    recover_mode.add_argument("--apply", action="store_true")

    reconcile = subparsers.add_parser(
        "reconcile-blocked-children",
        help="Dry-run or compare-and-swap exact dependency-failed children to pending",
    )
    reconcile.add_argument("--state", required=True)
    reconcile.add_argument(
        "--child-parent",
        action="append",
        type=_child_parent_pair,
        required=True,
        metavar="CHILD_TASK_ID=PARENT_TASK_ID",
    )
    reconcile_mode = reconcile.add_mutually_exclusive_group(required=True)
    reconcile_mode.add_argument("--dry-run", action="store_true")
    reconcile_mode.add_argument("--apply", action="store_true")

    status = subparsers.add_parser("run-status", help="Show exact durable task state")
    status.add_argument("--state", required=True)
    status.add_argument("--output")

    finalize = subparsers.add_parser(
        "finalize-run", help="Validate completion and write release indexes"
    )
    finalize.add_argument("plan")
    finalize.add_argument("--output-root", required=True)

    args = parser.parse_args(argv)
    if args.command == "coverage":
        _write_or_print(builtin_registry().inventory(), args.output)
    elif args.command == "interface":
        from .interface import serve_interface

        try:
            serve_interface(
                host=args.host,
                port=args.port,
                agentic_project=args.agentic_project,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    elif args.command == "runtime-check":
        report = runtime_check()
        _write_or_print(report, None)
        if not report["ready"]:
            raise SystemExit(1)
    elif args.command == "jev-decision":
        state_path = validated_cluster_path(args.state, must_exist=True)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            parser.error("OpenJev state must be a JSON object")
        client = OpenJevClient(endpoint=args.endpoint, model=args.model)
        decision = {
            "route": lambda: route_acquisition(state, client),
            "boolean": lambda: check_boolean(state, args.instructions, client),
            "urgency": lambda: score_urgency(state, client),
            "next_action": lambda: choose_next_action(state, client),
        }[args.kind]()
        _write_or_print(decision.model_dump(mode="json"), None)
    elif args.command == "hardware-profile":
        _write_or_print(detect_hardware_profile().model_dump(mode="json"), args.output)
    elif args.command == "prompt-policy":
        if args.request_file:
            request_path = validated_cluster_path(args.request_file, must_exist=True)
            request = request_path.read_text(encoding="utf-8")
        else:
            request = args.request
        profile = load_hardware_profile(args.hardware_profile)
        policy = compile_prompt_policy(request, profile)
        _write_or_print(policy.model_dump(mode="json"), args.output)
    elif args.command == "auto":
        from .autopilot import AutopilotError, run_autopilot

        if args.request_file:
            request = validated_cluster_path(args.request_file, must_exist=True).read_text(
                encoding="utf-8"
            )
        else:
            request = args.request
        try:
            report = run_autopilot(
                request,
                source_root=args.source_root,
                output_root=args.output_root,
                agentic_project=args.agentic_project,
                hardware_profile=load_hardware_profile(args.hardware_profile),
                signing_key=args.signing_key,
                freesurfer_license=args.freesurfer_license,
                execute=args.execute,
                gpu_workers=args.gpu_workers,
                max_pilot_subjects=args.max_pilot_subjects,
                allow_quarantine=args.allow_quarantine,
                allow_unsupported=args.allow_unsupported,
            )
        except AutopilotError as exc:
            _write_or_print(exc.report, None)
            raise SystemExit(f"autopilot stopped at {exc}") from exc
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        _write_or_print(report, None)
    elif args.command == "prompt-run":
        if args.request_file:
            request_path = validated_cluster_path(args.request_file, must_exist=True)
            request = request_path.read_text(encoding="utf-8")
        else:
            request = args.request
        try:
            result = prepare_prompt_run(
                request,
                source_root=args.source_root,
                output_root=args.output_root,
                manifest_path=args.manifest,
                dataset_id=args.dataset_id,
                adapter_id=args.adapter_id,
                identity_key=args.identity_key,
                agentic_project=args.agentic_project,
                hardware_profile=load_hardware_profile(args.hardware_profile),
                backend_evidence=args.backend_evidence,
                backend_evidence_key=args.backend_evidence_key,
                freesurfer_license=args.freesurfer_license,
                execute_optional=args.execute_optional,
                allow_quarantine=args.allow_quarantine,
                allow_unsupported=args.allow_unsupported,
                initialize=args.initialize,
                submit=args.submit,
                gpu_workers=args.gpu_workers,
                openjev_review=args.openjev_review,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        _write_or_print(result, None)
        if not result["ready"]:
            raise SystemExit(1)
    elif args.command == "backend-plan":
        policy = load_policy(args.policy)
        profile = load_hardware_profile(args.hardware_profile)
        evidence: dict[str, str] = {}
        if args.evidence:
            evidence_path = validated_cluster_path(args.evidence, must_exist=True)
            loaded = json.loads(evidence_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict) or any(
                value not in {"passed", "failed"} for value in loaded.values()
            ):
                parser.error("backend evidence must map test ids to passed or failed")
            evidence = {str(key): str(value) for key, value in loaded.items()}
        selected = plan_backends(policy, profile, evidence)
        _write_or_print(selected.model_dump(mode="json"), args.output)
        if not selected.ready:
            raise SystemExit(1)
    elif args.command == "qualification-evidence":
        plan = _load_execution_plan(args.plan)
        evidence = compile_backend_evidence(plan, RunState(plan.layout.state_path))
        _write_or_print(evidence, args.output)
    elif args.command == "build-backend-evidence":
        pilot_plan = validated_cluster_path(args.pilot_plan, must_exist=True)
        pilot_state = validated_cluster_path(args.pilot_state, must_exist=True)
        pilot_release = validated_cluster_path(args.pilot_release, must_exist=True)
        target_plan = validated_cluster_path(args.target_plan, must_exist=True)
        repository = validated_cluster_path(args.git_repository, must_exist=True)
        signing_key_path = validated_cluster_path(args.signing_key, must_exist=True)
        output = validated_cluster_path(args.output)
        document = build_backend_evidence(
            pilot_plan_path=pilot_plan,
            target_plan_path=target_plan,
            pilot_state_path=pilot_state,
            pilot_release_path=pilot_release,
            git_repository=repository,
            signing_key=read_signing_key(signing_key_path),
            key_id=args.key_id,
        )
        print(write_evidence_document(document, output))
    elif args.command == "create-backend-evidence-key":
        print(create_signing_key(args.output))
    elif args.command == "ingest-plan":
        if not args.identity_key or len(args.identity_key) < 12:
            parser.error("ingest-plan requires --identity-key or NEUROPREP_IDENTITY_KEY")
        manifest = load_manifest(args.manifest)
        _write_or_print(build_ingest_plan(manifest, args.identity_key), args.output)
    elif args.command == "executor-coverage":
        registry = builtin_registry()
        catalog = builtin_executor_catalog()
        missing = catalog.audit(registry)
        payload = {
            "ready": not missing,
            "catalog_digest": catalog.digest(),
            "executor_count": len(catalog.definitions),
            "missing_routes": missing,
            "executors": [item.model_dump(mode="json") for item in catalog.definitions],
        }
        _write_or_print(payload, args.output)
        if missing:
            raise SystemExit(1)
    elif args.command == "compile-run":
        if not args.identity_key or len(args.identity_key) < 12:
            parser.error("compile-run requires a strong identity key")
        manifest = load_manifest(args.manifest)
        graph, logical_plan = build_graph_and_plan(manifest, args.identity_key)
        if graph.quarantined and not args.allow_quarantine:
            parser.error(
                f"{len(graph.quarantined)} acquisitions are quarantined; review before execution"
            )
        if graph.unsupported and not args.allow_unsupported:
            parser.error(
                f"{len(graph.unsupported)} acquisitions are unsupported; add a plugin first"
            )
        registry = builtin_registry()
        catalog = builtin_executor_catalog()
        missing = catalog.audit(registry)
        if missing:
            parser.error("executor catalog gaps: " + ", ".join(missing))
        try:
            agentic_project = validated_agentic_project(args.agentic_project)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        layout = RunLayout(
            dataset_id=manifest.dataset.dataset_id,
            source_root=args.source_root,
            bids_root=args.bids_root,
            derivatives_root=args.derivatives_root,
            work_root=args.work_root,
            state_path=args.state_path,
            agentic_project=str(agentic_project),
            freesurfer_license=args.freesurfer_license,
            execute_optional=args.execute_optional,
        )
        policy = load_policy(args.policy) if args.policy else RunPolicy()
        profile = load_hardware_profile(args.hardware_profile)
        probe_plan = compile_execution_plan(
            graph,
            logical_plan,
            layout,
            catalog,
            run_policy=policy.model_dump(mode="json"),
            policy_digest=policy.digest(),
            backend_decisions=[],
            planned_smoke_tests=[],
        )
        required_executor_ids = {task.executor_id for task in probe_plan.tasks}
        unqualified = plan_backends(
            policy,
            profile,
            {},
            required_executor_ids=required_executor_ids,
        )
        target_smoke_tests = [
            item.model_dump(mode="json") for item in unqualified.smoke_tests
        ]
        evidence: dict[str, str] = {}
        evidence_binding = None
        if args.backend_evidence:
            signing_key = (
                read_signing_key(args.backend_evidence_key)
                if args.backend_evidence_key
                else None
            )
            loaded_evidence = validate_backend_evidence(
                evidence_path=args.backend_evidence,
                signing_key=signing_key,
                target_plan=probe_plan.model_dump(mode="json"),
                target_smoke_tests=target_smoke_tests,
                catalog_digest=catalog.digest(),
                target_git_commit=current_git_commit(layout.agentic_project),
            )
            evidence = dict(loaded_evidence.statuses)
            evidence_binding = loaded_evidence.binding
        selected = plan_backends(
            policy,
            profile,
            evidence,
            required_executor_ids=required_executor_ids,
            backend_evidence=evidence_binding,
        )
        plan = compile_execution_plan(
            graph,
            logical_plan,
            layout,
            catalog,
            run_policy=policy.model_dump(mode="json"),
            policy_digest=policy.digest(),
            backend_decisions=[item.model_dump(mode="json") for item in selected.decisions],
            backend_qualification_tests=target_smoke_tests,
            planned_smoke_tests=[item.model_dump(mode="json") for item in selected.smoke_tests],
            backend_evidence=evidence_binding,
        )
        print(save_execution_plan(plan, args.output))
    elif args.command == "preflight-run":
        plan = _load_execution_plan(args.plan)
        report = preflight_execution_plan(
            plan,
            builtin_executor_catalog(),
            check_inputs=not args.skip_input_check,
        )
        _write_or_print(report.model_dump(mode="json"), args.output)
        if not report.ready:
            raise SystemExit(1)
    elif args.command == "prefetch-images":
        plan = _load_execution_plan(args.plan)
        catalog = builtin_executor_catalog()
        result = prefetch_images(plan, catalog) if args.execute else image_inventory(plan, catalog)
        _write_or_print(result, args.output)
    elif args.command == "catalog-images":
        catalog = builtin_executor_catalog()
        result = (
            prefetch_catalog_images(catalog)
            if args.execute
            else catalog_image_inventory(catalog, inspect=args.inspect)
        )
        _write_or_print(result, args.output)
    elif args.command == "prepare-assets":
        plan = _load_execution_plan(args.plan)
        result = prepare_runtime_assets(plan) if args.execute else runtime_asset_inventory(plan)
        _write_or_print(result, args.output)
    elif args.command == "run-init":
        plan = _load_execution_plan(args.plan)
        qualification_required = bool(
            plan.backend_qualification_tests or plan.planned_smoke_tests
        )
        try:
            revalidate_execution_evidence(
                plan=plan.model_dump(mode="json"),
                signing_key=(
                    read_signing_key(args.backend_evidence_key)
                    if args.backend_evidence_key
                    else None
                ),
                target_git_commit=(
                    current_git_commit(plan.layout.agentic_project)
                    if qualification_required
                    else ""
                ),
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        report = preflight_execution_plan(plan, builtin_executor_catalog())
        if not report.ready and not args.allow_incomplete_preflight:
            print(report.model_dump_json(indent=2))
            raise SystemExit(1)
        RunState(plan.layout.state_path).initialize(plan, args.plan)
        _write_or_print(RunState(plan.layout.state_path).summary(), None)
    elif args.command == "recover-stale-task":
        result = RunState(args.state).recover_stale_task(
            args.task_id,
            expected_worker_id=args.expected_worker_id,
            expected_attempt=args.expected_attempt,
            expected_lease_expires_at=args.expected_lease_expires_at,
            finished_at=args.finished_at,
            reason=args.reason,
            log_path=args.log_path,
            return_code=args.return_code,
            apply=args.apply,
        )
        _write_or_print(result, None)
    elif args.command == "reconcile-blocked-children":
        pairs: dict[str, str] = {}
        for child, parent in args.child_parent:
            if child in pairs:
                parser.error(f"duplicate child task id: {child}")
            pairs[child] = parent
        result = RunState(args.state).reconcile_blocked_children(
            pairs,
            apply=args.apply,
        )
        _write_or_print(result, None)
    elif args.command == "worker":
        summary = run_worker(
            args.state,
            args.worker_id,
            set(args.resource_class),
            lease_seconds=args.lease_seconds,
            max_tasks=args.max_tasks,
            exit_when_idle=not args.stay_alive,
        )
        _write_or_print(summary, None)
        terminal = RunState(args.state).summary()["counts"]
        if terminal.get("failed", 0) or terminal.get("blocked", 0):
            raise SystemExit(1)
    elif args.command == "run-status":
        _write_or_print(RunState(args.state).summary(), args.output)
    elif args.command == "finalize-run":
        plan = _load_execution_plan(args.plan)
        summary = finalize_execution_run(
            plan,
            RunState(plan.layout.state_path),
            args.output_root,
        )
        _write_or_print(summary, None)
    elif args.command == "crew":
        run_control_flow(args.workflow, args.request)
    elif args.command == "schemas":
        from .models import CanonicalAcquisitionGraph, CompletionContract, PlanResult
        from .policy import BackendPlan, HardwareProfile
        from .services import ManifestDocument

        payload = {
            "manifest": ManifestDocument.model_json_schema(),
            "acquisition_graph": CanonicalAcquisitionGraph.model_json_schema(),
            "plan": PlanResult.model_json_schema(),
            "completion_contract": CompletionContract.model_json_schema(),
            "run_policy": RunPolicy.model_json_schema(),
            "hardware_profile": HardwareProfile.model_json_schema(),
            "backend_plan": BackendPlan.model_json_schema(),
            "backend_evidence": BackendEvidenceDocument.model_json_schema(),
            "prompt_path_binding": PromptPathBinding.model_json_schema(),
        }
        _write_or_print(payload, args.output)
    elif args.command == "validate-control-artifact":
        artifact = validated_cluster_path(args.artifact, must_exist=True)
        print(validate_control_artifact(str(artifact), args.workflow))


if __name__ == "__main__":
    cli()
