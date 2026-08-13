from __future__ import annotations

import argparse
import json
import os

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
from .policy import (
    RunPolicy,
    compile_prompt_policy,
    detect_hardware_profile,
    load_hardware_profile,
    load_policy,
    plan_backends,
)
from .preflight import preflight_execution_plan
from .qualification import compile_backend_evidence
from .registry import builtin_registry
from .release import finalize_execution_run
from .runtime import run_worker
from .runtime_assets import prepare_runtime_assets, runtime_asset_inventory
from .services import build_graph_and_plan, build_ingest_plan, load_manifest
from .settings import validated_cluster_path
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


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset-agnostic multimodal NeuroPrep control plane"
    )
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
    compile_run.add_argument("--freesurfer-license")
    compile_run.add_argument("--execute-optional", action="store_true")
    compile_run.add_argument("--policy")
    compile_run.add_argument("--hardware-profile")
    compile_run.add_argument("--backend-evidence")
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

    status = subparsers.add_parser("run-status", help="Show exact durable task state")
    status.add_argument("--state", required=True)
    status.add_argument("--output")

    finalize = subparsers.add_parser(
        "finalize-run", help="Validate completion and write release indexes"
    )
    finalize.add_argument("plan")
    finalize.add_argument("--output-root", required=True)

    args = parser.parse_args()
    if args.command == "coverage":
        _write_or_print(builtin_registry().inventory(), args.output)
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
        layout = RunLayout(
            dataset_id=manifest.dataset.dataset_id,
            source_root=args.source_root,
            bids_root=args.bids_root,
            derivatives_root=args.derivatives_root,
            work_root=args.work_root,
            state_path=args.state_path,
            freesurfer_license=args.freesurfer_license,
            execute_optional=args.execute_optional,
        )
        policy = load_policy(args.policy) if args.policy else RunPolicy()
        profile = load_hardware_profile(args.hardware_profile)
        evidence: dict[str, str] = {}
        if args.backend_evidence:
            evidence_path = validated_cluster_path(args.backend_evidence, must_exist=True)
            loaded = json.loads(evidence_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict) or any(
                value not in {"passed", "failed"} for value in loaded.values()
            ):
                parser.error("backend evidence must map test ids to passed or failed")
            evidence = {str(key): str(value) for key, value in loaded.items()}
        selected = plan_backends(policy, profile, evidence)
        plan = compile_execution_plan(
            graph,
            logical_plan,
            layout,
            catalog,
            run_policy=policy.model_dump(mode="json"),
            policy_digest=policy.digest(),
            backend_decisions=[item.model_dump(mode="json") for item in selected.decisions],
            planned_smoke_tests=[item.model_dump(mode="json") for item in selected.smoke_tests],
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
        report = preflight_execution_plan(plan, builtin_executor_catalog())
        if not report.ready and not args.allow_incomplete_preflight:
            print(report.model_dump_json(indent=2))
            raise SystemExit(1)
        RunState(plan.layout.state_path).initialize(plan, args.plan)
        _write_or_print(RunState(plan.layout.state_path).summary(), None)
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
        }
        _write_or_print(payload, args.output)
    elif args.command == "validate-control-artifact":
        artifact = validated_cluster_path(args.artifact, must_exist=True)
        print(validate_control_artifact(str(artifact), args.workflow))


if __name__ == "__main__":
    cli()
