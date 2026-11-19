from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from .alignment import (
    align_dataset,
    align_visit,
    create_common_grid_reference,
    validate_common_grid,
)
from .anatomical import finalize_anatomical_subject
from .assembly import DEFAULT_MODALITIES, assemble_visits
from .autoconfigure import autoconfigure_dataset
from .config import ConfigError, load_dataset_configs, select_dataset
from .contracts import audit_inventory
from .execution import execute_plan, load_plan, render_commands, write_nextflow_tasks
from .engine import TaskDatabase, run_worker, write_slurm_pool
from .experts import ExpertRegistry
from .finalization import finalize_modality_subject
from .grouping import group_inventory, manifest_summary, write_manifest
from .harmonization import HarmonizationSpec, harmonize_folds, harmonize_table
from .inventory import inventory_dataset
from .model_ready import build_subject_model_ready_manifest, index_model_ready_subjects
from .planner import build_plan
from .profiles import DEFAULT_EXECUTION_PROFILE, PROFILES
from .phenotypes import create_grouped_splits, join_phenotypes
from .production import (
    forecast_plan,
    load_test_jsonl,
    monitor_engine,
    production_audit,
    verify_container_lock,
    write_exclusion_report,
)
from .quality import production_qc
from .qc import aggregate_qc_reports, inspect_image, write_qc, write_qc_decision
from .release import write_release_index
from .registration import index_transform_files
from .registry import PipelineRegistry
from .staging import (
    stage_selection,
    stage_visit,
    write_subject_list,
    write_visit_execution_index,
)
from .tensors import (
    TensorPolicy,
    index_tensor_manifests,
    prepare_dataset_tensors,
    prepare_visit_tensors,
)
from .upstream import (
    preprocess_sensor_subject,
    preprocess_spatial_modalities,
    preprocess_susceptibility_subject,
)


def _write_json(payload: dict, path: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _dataset(args: argparse.Namespace):
    dataset = select_dataset(load_dataset_configs(args.config), args.dataset_id)
    if getattr(args, "bids_view", False):
        return replace(
            dataset,
            root=dataset.bids_root,
            adapter="bids",
            input_layout="bids",
        )
    return dataset


def command_inventory(args: argparse.Namespace) -> int:
    inventory = inventory_dataset(_dataset(args))
    _write_json(inventory.to_dict(), args.output)
    return 0


def command_plan(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    inventory = inventory_dataset(dataset)
    registry = (
        PipelineRegistry.from_path(args.registry) if args.registry else PipelineRegistry.default()
    )
    plan = build_plan(dataset, inventory, registry)
    _write_json(plan.to_dict(), args.output)
    return 0


def command_audit(args: argparse.Namespace) -> int:
    report = audit_inventory(inventory_dataset(_dataset(args)), max_examples=args.max_examples)
    _write_json(report, args.output)
    return 1 if report["severity_counts"].get("error", 0) else 0


def _command_manifest(args: argparse.Namespace, apply_selection: bool) -> int:
    dataset = _dataset(args)
    rows = group_inventory(
        inventory_dataset(dataset),
        dataset,
        apply_selection=apply_selection,
    )
    write_manifest(rows, args.output, dataset.dataset_id)
    summary = manifest_summary(rows, dataset.dataset_id)
    if args.summary:
        _write_json(summary, args.summary)
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if apply_selection and summary["selection_counts"].get("manual-review", 0) else 0


def command_group(args: argparse.Namespace) -> int:
    return _command_manifest(args, apply_selection=False)


def command_select(args: argparse.Namespace) -> int:
    return _command_manifest(args, apply_selection=True)


def command_stage(args: argparse.Namespace) -> int:
    summary = stage_selection(
        selection_path=args.selection,
        output_root=args.output,
        modalities=args.modalities,
        roles=args.roles,
        execute=args.execute,
        link_mode=args.link_mode,
        dataset_name=args.dataset_name,
        primary_only_modalities=args.primary_only_modalities,
    )
    _write_json(summary, args.summary)
    return 0


def command_stage_visit(args: argparse.Namespace) -> int:
    summary = stage_visit(
        bids_root=args.bids_root,
        output_root=args.output,
        subject=args.subject,
        visit_id=args.visit_id,
        marker=args.marker,
        tolerance_days=args.tolerance_days,
        link_mode=args.link_mode,
        primary_t1=args.primary_t1,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_visit_execution_index(args: argparse.Namespace) -> int:
    summary = write_visit_execution_index(
        roots=args.roots,
        marker=args.marker,
        subject=args.subject,
        pipeline=args.pipeline,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_subjects(args: argparse.Namespace) -> int:
    count = write_subject_list(args.bids_root, args.datatype, args.output)
    print(f"Wrote {count} subjects to {args.output}")
    return 0


def command_render(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    lines = render_commands(plan)
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def command_run(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    results = execute_plan(
        plan,
        state_path=args.state,
        dry_run=not args.execute,
        allow_custom_gaps=args.allow_custom_gaps,
    )
    for result in results:
        detail = f" - {result.message}" if result.message else ""
        print(f"{result.status:8} {result.task_id}{detail}")
    return 1 if any(result.status in {"failed", "blocked"} for result in results) else 0


def command_export_nextflow(args: argparse.Namespace) -> int:
    count = write_nextflow_tasks(
        load_plan(args.plan),
        args.output,
        allow_custom_gaps=args.allow_custom_gaps,
    )
    print(f"Wrote {count} runnable tasks to {args.output}")
    return 0


def command_qc(args: argparse.Namespace) -> int:
    reports = [inspect_image(path) for path in args.images]
    if args.output:
        write_qc(reports, args.output)
    else:
        _write_json({"images": [report.to_dict() for report in reports]}, None)
    return 1 if any(report.status != "pass" for report in reports) else 0


def command_doctor(_: argparse.Namespace) -> int:
    missing = []
    container_runtime = shutil.which("apptainer") or shutil.which("singularity")
    print(f"{'container runtime':32} {container_runtime or 'MISSING'}")
    if container_runtime is None:
        missing.append("apptainer/singularity")
    nextflow = shutil.which("nextflow")
    print(f"{'nextflow':32} {nextflow or 'MISSING'}")
    if nextflow is None:
        missing.append("nextflow")
    for tool in ("dcm2niix", "antsRegistrationSyNQuick.sh", "antsApplyTransforms"):
        location = shutil.which(tool)
        print(f"{tool:32} {location or 'containerized'}")
    return 1 if missing else 0


def command_experts(_: argparse.Namespace) -> int:
    _write_json({"experts": ExpertRegistry().describe()}, None)
    return 0


def command_profiles(_: argparse.Namespace) -> int:
    _write_json(
        {
            "default": DEFAULT_EXECUTION_PROFILE,
            "profiles": {
                name: profile.to_dict() for name, profile in sorted(PROFILES.items())
            },
        },
        None,
    )
    return 0


def command_engine_init(args: argparse.Namespace) -> int:
    count = TaskDatabase(args.database).initialize(load_plan(args.plan))
    print(f"Initialized {count} tasks in {args.database}")
    return 0


def command_engine_retry(args: argparse.Namespace) -> int:
    report = TaskDatabase(args.database).retry_failed(args.task_id)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_engine_recover(args: argparse.Namespace) -> int:
    report = TaskDatabase(args.database).recover_workers(
        args.worker_prefix,
        reason=args.reason,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_worker(args: argparse.Namespace) -> int:
    result = run_worker(
        database=args.database,
        work_root=args.work_root,
        resource_class=args.resource_class,
        worker_id=args.worker_id,
        max_tasks=args.max_tasks,
        poll_seconds=args.poll_seconds,
        idle_exit_seconds=args.idle_exit_seconds,
        lease_seconds=args.lease_seconds,
    )
    print(
        f"worker={result.worker_id} complete={result.completed} "
        f"failed={result.failed} blocked={result.blocked}"
    )
    return 1 if result.failed else 0


def command_status(args: argparse.Namespace) -> int:
    summary = TaskDatabase(args.database).summary()
    _write_json(summary, args.output)
    return (
        1
        if summary["status_counts"].get("failed", 0) or summary["status_counts"].get("blocked", 0)
        else 0
    )


def command_export_slurm_pool(args: argparse.Namespace) -> int:
    store = TaskDatabase(args.database)
    store.initialize(load_plan(args.plan))
    paths = write_slurm_pool(
        output_dir=args.output,
        database=args.database,
        work_root=args.work_root,
        cpu_workers=args.cpu_workers,
        gpu_workers=args.gpu_workers,
        cpu_partition=args.cpu_partition,
        gpu_partition=args.gpu_partition,
        cpu_cpus=args.cpu_cpus,
        cpu_memory_gb=args.cpu_memory_gb,
        gpu_cpus=args.gpu_cpus,
        gpu_memory_gb=args.gpu_memory_gb,
    )
    print("\n".join(str(path) for path in paths))
    return 0


def command_qc_aggregate(args: argparse.Namespace) -> int:
    decision = aggregate_qc_reports(args.reports, required=args.required)
    write_qc_decision(decision, args.output)
    return 0 if decision.status == "pass" else 1


def command_release_index(args: argparse.Namespace) -> int:
    summary = write_release_index(
        args.database,
        args.output,
        checksum_files=args.checksum_files,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["release_ready"] else 1


def command_harmonize(args: argparse.Namespace) -> int:
    report = harmonize_table(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        report_path=args.report,
        spec=HarmonizationSpec(
            participant_column=args.participant_column,
            split_column=args.split_column,
            batch_column=args.batch_column,
            protected_columns=tuple(args.protected_columns),
            feature_columns=tuple(args.feature_columns),
            train_label=args.train_label,
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_capture(args: argparse.Namespace) -> int:
    argv = list(args.argv)
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        raise ValueError("capture requires a command after --")
    completed = subprocess.run(argv, capture_output=True, check=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(completed.stdout)
    temporary.replace(args.output)
    if completed.stderr:
        sys.stderr.buffer.write(completed.stderr)
    return completed.returncode


def command_index_transforms(args: argparse.Namespace) -> int:
    graph = index_transform_files(args.subject, args.search, args.output)
    print(f"Indexed {len(graph.records)} transforms in {args.output}")
    return 0


def command_finalize_anatomical(args: argparse.Namespace) -> int:
    summary = finalize_anatomical_subject(
        preprocessing_root=args.pipeline_root,
        output_root=args.output_root,
        subject=args.subject,
        marker=args.marker,
        required_spaces=tuple(args.require_space),
        source_pipeline=args.source_pipeline,
        registration_container=args.registration_container,
        standard_template=args.standard_template,
        standard_mask=args.standard_mask,
        threads=args.threads,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


def command_finalize_modality(args: argparse.Namespace) -> int:
    summary = finalize_modality_subject(
        pipeline_root=args.pipeline_root,
        output_root=args.output_root,
        bids_root=args.bids_root,
        subject=args.subject,
        modality=args.modality,
        marker=args.marker,
        required_spaces=tuple(args.require_space),
        support_roots=tuple(args.support_root),
        source_pipeline=args.source_pipeline,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


def command_model_ready_subject(args: argparse.Namespace) -> int:
    manifest = build_subject_model_ready_manifest(
        dataset_id=args.dataset_id,
        subject=args.subject,
        finalization_markers=args.finalization_markers,
        qc_report=args.qc_report,
        output=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "pass" else 1


def command_model_ready_index(args: argparse.Namespace) -> int:
    summary = index_model_ready_subjects(args.root, args.output_jsonl, args.output_tsv)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


def command_autoconfigure(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    summary = autoconfigure_dataset(
        inventory_dataset(dataset),
        dataset,
        args.output,
        minimum_confidence=args.minimum_confidence,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


def command_preprocess_spatial(args: argparse.Namespace) -> int:
    summary = preprocess_spatial_modalities(
        bids_root=args.bids_root,
        output_root=args.output_root,
        work_root=args.work_root,
        subject=args.subject,
        modalities=tuple(args.modalities),
        reference_root=args.reference_root,
        template=args.template,
        execute=args.execute,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] in {"pass", "planned"} else 1


def command_preprocess_sensor(args: argparse.Namespace) -> int:
    summary = preprocess_sensor_subject(
        bids_root=args.bids_root,
        output_root=args.output_root,
        subject=args.subject,
        modality=args.modality,
        session=args.session,
        low_frequency=args.low_frequency,
        high_frequency=args.high_frequency,
        resample_hz=args.resample_hz,
        apply_ica=not args.no_ica,
        execute=args.execute,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] in {"pass", "planned"} else 1


def command_preprocess_susceptibility(args: argparse.Namespace) -> int:
    summary = preprocess_susceptibility_subject(
        bids_root=args.bids_root,
        output_root=args.output_root,
        work_root=args.work_root,
        subject=args.subject,
        reference_root=args.reference_root,
        template=args.template,
        execute=args.execute,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] in {"pass", "planned"} else 1


def command_assemble_visits(args: argparse.Namespace) -> int:
    summary = assemble_visits(
        sample_index=args.sample_index,
        output_root=args.output_root,
        expected_modalities=tuple(args.expected_modalities),
        tolerance_days=args.tolerance_days,
        bids_root=args.bids_root,
        dataset_id=args.dataset_id,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


def command_validate_grid(args: argparse.Namespace) -> int:
    report = validate_common_grid(args.images, args.reference, args.affine_tolerance)
    _write_json(report, args.output)
    return 0 if report["status"] == "pass" else 1


def command_grid_reference(args: argparse.Namespace) -> int:
    report = create_common_grid_reference(
        args.sample_index,
        args.output,
        tuple(args.preferred_modalities),
        args.required_space,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_align_visit(args: argparse.Namespace) -> int:
    report = align_visit(
        visit_manifest=args.visit_manifest,
        reference=args.reference,
        output_root=args.output_root,
        sensor_transform_root=args.sensor_transform_root,
        require_standard_sensor_coordinates=args.require_standard_sensor_coordinates,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


def command_align_dataset(args: argparse.Namespace) -> int:
    report = align_dataset(
        visit_index=args.visit_index,
        reference=args.reference,
        output_root=args.output_root,
        output_index=args.output_index,
        sensor_transform_root=args.sensor_transform_root,
        require_standard_sensor_coordinates=args.require_standard_sensor_coordinates,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


def _tensor_policy(args: argparse.Namespace) -> TensorPolicy:
    return TensorPolicy(
        target_shape=tuple(args.target_shape),
        image_modalities=tuple(args.image_modalities),
        fmri_window=args.fmri_window,
        fmri_stride=args.fmri_stride,
        sensor_window_seconds=args.sensor_window_seconds,
        sensor_stride_seconds=args.sensor_stride_seconds,
        dwi_shells=tuple(args.dwi_shells),
    )


def command_tensorize_visit(args: argparse.Namespace) -> int:
    report = prepare_visit_tensors(
        args.aligned_manifest,
        args.output_root,
        _tensor_policy(args),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


def command_tensor_index(args: argparse.Namespace) -> int:
    report = index_tensor_manifests(args.root, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


def command_tensorize_dataset(args: argparse.Namespace) -> int:
    report = prepare_dataset_tensors(
        args.aligned_index,
        args.output_root,
        args.output_index,
        _tensor_policy(args),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


def command_production_qc(args: argparse.Namespace) -> int:
    report = production_qc(
        args.finalization_markers,
        args.output,
        args.thresholds,
        review_policy=args.review_policy,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"pass", "partial"} else 1


def command_phenotype_join(args: argparse.Namespace) -> int:
    report = join_phenotypes(
        visit_index=args.visit_index,
        phenotype_table=args.phenotype_table,
        output=args.output,
        report=args.report,
        participant_column=args.participant_column,
        session_column=args.session_column,
        date_column=args.date_column,
        label_columns=tuple(args.label_columns),
        covariate_columns=tuple(args.covariate_columns),
        tolerance_days=args.tolerance_days,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


def command_make_splits(args: argparse.Namespace) -> int:
    report = create_grouped_splits(
        joined_index=args.joined_index,
        output=args.output,
        report=args.report,
        folds=args.folds,
        seed=args.seed,
        site_column=args.site_column,
        label_column=args.label_column,
        strategy=args.strategy,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_harmonize_folds(args: argparse.Namespace) -> int:
    report = harmonize_folds(
        input_path=args.input,
        output_path=args.output,
        output_root=args.output_root,
        report_path=args.report,
        spec=HarmonizationSpec(
            participant_column=args.participant_column,
            split_column=args.split_column,
            batch_column=args.batch_column,
            protected_columns=tuple(args.protected_columns),
            feature_columns=tuple(args.feature_columns),
            train_label=args.train_label,
            allow_repeated_participants=True,
        ),
        fold_column=args.fold_column,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_verify_containers(args: argparse.Namespace) -> int:
    report = verify_container_lock(args.lock)
    _write_json(report, args.output)
    return 0 if report["status"] == "pass" else 1


def command_forecast(args: argparse.Namespace) -> int:
    report = forecast_plan(args.plan, args.output)
    if not args.output:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_monitor(args: argparse.Namespace) -> int:
    report = monitor_engine(args.database, args.output)
    if not args.output:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report.get("status_counts", {}).get("failed") else 0


def command_exclusions(args: argparse.Namespace) -> int:
    report = write_exclusion_report(args.root, args.output_tsv, args.output_json)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_load_test(args: argparse.Namespace) -> int:
    report = load_test_jsonl(args.input, args.repetitions, args.minimum_rows_per_second)
    _write_json(report, args.output)
    return 0 if report["status"] == "pass" else 1


def command_production_audit(args: argparse.Namespace) -> int:
    report = production_audit(
        args.plan,
        args.container_lock,
        args.validation_root,
        args.output,
        tuple(args.required_modalities),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neuroprep")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, handler in (
        ("inventory", command_inventory),
        ("audit", command_audit),
        ("group", command_group),
        ("select", command_select),
        ("plan", command_plan),
    ):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--dataset-id")
        subparser.add_argument(
            "--bids-view",
            action="store_true",
            help="Operate on bids_root after a legacy/DICOM ingest instead of the source root",
        )
        subparser.add_argument(
            "--output",
            type=Path,
            required=name in {"group", "select"},
        )
        if name == "plan":
            subparser.add_argument("--registry", type=Path)
        if name == "audit":
            subparser.add_argument("--max-examples", type=int, default=50)
        if name in {"group", "select"}:
            subparser.add_argument("--summary", type=Path)
        subparser.set_defaults(handler=handler)

    render = subparsers.add_parser("render")
    render.add_argument("--plan", type=Path, required=True)
    render.add_argument("--output", type=Path)
    render.set_defaults(handler=command_render)

    stage = subparsers.add_parser("stage")
    stage.add_argument("--selection", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--modalities", nargs="+", required=True)
    stage.add_argument(
        "--roles",
        nargs="+",
        default=["primary-reference", "include"],
    )
    stage.add_argument("--link-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    stage.add_argument("--dataset-name", default="NeuroPrep selected processing view")
    stage.add_argument(
        "--primary-only-modalities",
        nargs="*",
        default=[],
        help="For listed modalities, stage only the selected primary-reference acquisition",
    )
    stage.add_argument("--execute", action="store_true", help="Materialize the staged BIDS view")
    stage.add_argument("--summary", type=Path)
    stage.set_defaults(handler=command_stage)

    stage_visit_parser = subparsers.add_parser("stage-visit")
    stage_visit_parser.add_argument("--bids-root", type=Path, required=True)
    stage_visit_parser.add_argument("--output", type=Path, required=True)
    stage_visit_parser.add_argument("--subject", required=True)
    stage_visit_parser.add_argument("--visit-id", required=True)
    stage_visit_parser.add_argument("--marker", type=Path, required=True)
    stage_visit_parser.add_argument("--tolerance-days", type=int, default=30)
    stage_visit_parser.add_argument("--primary-t1", type=Path)
    stage_visit_parser.add_argument(
        "--link-mode", choices=("hardlink", "symlink", "copy"), default="hardlink"
    )
    stage_visit_parser.set_defaults(handler=command_stage_visit)

    visit_index = subparsers.add_parser("visit-execution-index")
    visit_index.add_argument("--roots", type=Path, nargs="+", required=True)
    visit_index.add_argument("--marker", type=Path, required=True)
    visit_index.add_argument("--subject", required=True)
    visit_index.add_argument("--pipeline", required=True)
    visit_index.set_defaults(handler=command_visit_execution_index)

    subjects = subparsers.add_parser("subjects")
    subjects.add_argument("--bids-root", type=Path, required=True)
    subjects.add_argument("--datatype", required=True)
    subjects.add_argument("--output", type=Path, required=True)
    subjects.set_defaults(handler=command_subjects)

    export = subparsers.add_parser("export-nextflow")
    export.add_argument("--plan", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument(
        "--allow-custom-gaps",
        action="store_true",
        help="Omit unconfigured custom tasks (unsafe for production execution)",
    )
    export.set_defaults(handler=command_export_nextflow)

    run = subparsers.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--state", type=Path, required=True)
    run.add_argument("--execute", action="store_true", help="Execute commands; default is dry-run")
    run.add_argument("--allow-custom-gaps", action="store_true")
    run.set_defaults(handler=command_run)

    qc = subparsers.add_parser("qc")
    qc.add_argument("images", type=Path, nargs="+")
    qc.add_argument("--output", type=Path)
    qc.set_defaults(handler=command_qc)

    doctor = subparsers.add_parser("doctor")
    doctor.set_defaults(handler=command_doctor)
    experts = subparsers.add_parser("experts")
    experts.set_defaults(handler=command_experts)
    profiles = subparsers.add_parser("profiles")
    profiles.set_defaults(handler=command_profiles)

    engine_init = subparsers.add_parser("engine-init")
    engine_init.add_argument("--plan", type=Path, required=True)
    engine_init.add_argument("--database", type=Path, required=True)
    engine_init.set_defaults(handler=command_engine_init)

    engine_retry = subparsers.add_parser("engine-retry")
    engine_retry.add_argument("--database", type=Path, required=True)
    engine_retry.add_argument("--task-id", nargs="+", required=True)
    engine_retry.set_defaults(handler=command_engine_retry)

    engine_recover = subparsers.add_parser("engine-recover")
    engine_recover.add_argument("--database", type=Path, required=True)
    engine_recover.add_argument("--worker-prefix", nargs="+", required=True)
    engine_recover.add_argument(
        "--reason",
        default="Worker pool was intentionally stopped",
    )
    engine_recover.set_defaults(handler=command_engine_recover)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--database", type=Path, required=True)
    worker.add_argument("--work-root", type=Path, required=True)
    worker.add_argument("--resource-class", choices=("cpu", "gpu", "any"), required=True)
    worker.add_argument("--worker-id")
    worker.add_argument("--max-tasks", type=int, default=0)
    worker.add_argument("--poll-seconds", type=float, default=5.0)
    worker.add_argument("--idle-exit-seconds", type=float, default=0.0)
    worker.add_argument("--lease-seconds", type=int, default=300)
    worker.set_defaults(handler=command_worker)

    status = subparsers.add_parser("status")
    status.add_argument("--database", type=Path, required=True)
    status.add_argument("--output", type=Path)
    status.set_defaults(handler=command_status)

    pool = subparsers.add_parser("export-slurm-pool")
    pool.add_argument("--plan", type=Path, required=True)
    pool.add_argument("--database", type=Path, required=True)
    pool.add_argument("--work-root", type=Path, required=True)
    pool.add_argument("--output", type=Path, required=True)
    pool.add_argument("--cpu-workers", type=int, default=2)
    pool.add_argument("--gpu-workers", type=int, default=1)
    pool.add_argument("--cpu-partition", default="cscc-cpu-p")
    pool.add_argument("--gpu-partition", default="cscc-gpu-p")
    pool.add_argument("--cpu-cpus", type=int, default=16)
    pool.add_argument("--cpu-memory-gb", type=int, default=64)
    pool.add_argument("--gpu-cpus", type=int, default=16)
    pool.add_argument("--gpu-memory-gb", type=int, default=64)
    pool.set_defaults(handler=command_export_slurm_pool)

    qc_aggregate = subparsers.add_parser("qc-aggregate")
    qc_aggregate.add_argument("reports", nargs="+", type=Path)
    qc_aggregate.add_argument("--required", nargs="*", default=[])
    qc_aggregate.add_argument("--output", type=Path, required=True)
    qc_aggregate.set_defaults(handler=command_qc_aggregate)

    release = subparsers.add_parser("release-index")
    release.add_argument("--database", type=Path, required=True)
    release.add_argument("--output", type=Path, required=True)
    release.add_argument("--checksum-files", action="store_true")
    release.set_defaults(handler=command_release_index)

    harmonize = subparsers.add_parser("harmonize")
    harmonize.add_argument("--input", type=Path, required=True)
    harmonize.add_argument("--output", type=Path, required=True)
    harmonize.add_argument("--model", type=Path, required=True)
    harmonize.add_argument("--report", type=Path, required=True)
    harmonize.add_argument("--participant-column", default="participant_id")
    harmonize.add_argument("--split-column", default="split")
    harmonize.add_argument("--batch-column", default="site")
    harmonize.add_argument("--protected-columns", nargs="*", default=[])
    harmonize.add_argument("--feature-columns", nargs="+", required=True)
    harmonize.add_argument("--train-label", default="train")
    harmonize.set_defaults(handler=command_harmonize)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("argv", nargs=argparse.REMAINDER)
    capture.set_defaults(handler=command_capture)

    transforms = subparsers.add_parser("index-transforms")
    transforms.add_argument("--subject", required=True)
    transforms.add_argument("--search", nargs="+", type=Path, required=True)
    transforms.add_argument("--output", type=Path, required=True)
    transforms.set_defaults(handler=command_index_transforms)

    finalize_anatomical = subparsers.add_parser("finalize-anatomical")
    finalize_anatomical.add_argument(
        "--pipeline-root", "--smriprep-root", dest="pipeline_root", type=Path, required=True
    )
    finalize_anatomical.add_argument(
        "--source-pipeline", choices=("smriprep", "deepprep"), default="smriprep"
    )
    finalize_anatomical.add_argument("--output-root", type=Path, required=True)
    finalize_anatomical.add_argument("--subject", required=True)
    finalize_anatomical.add_argument("--marker", type=Path, required=True)
    finalize_anatomical.add_argument("--require-space", action="append", default=[])
    finalize_anatomical.add_argument("--registration-container", type=Path)
    finalize_anatomical.add_argument("--standard-template", type=Path)
    finalize_anatomical.add_argument("--standard-mask", type=Path)
    finalize_anatomical.add_argument("--threads", type=int, default=8)
    finalize_anatomical.set_defaults(handler=command_finalize_anatomical)

    finalize_modality = subparsers.add_parser("finalize-modality")
    finalize_modality.add_argument("--pipeline-root", type=Path, required=True)
    finalize_modality.add_argument("--output-root", type=Path, required=True)
    finalize_modality.add_argument("--bids-root", type=Path, required=True)
    finalize_modality.add_argument("--subject", required=True)
    finalize_modality.add_argument("--modality", required=True)
    finalize_modality.add_argument("--source-pipeline")
    finalize_modality.add_argument("--marker", type=Path, required=True)
    finalize_modality.add_argument("--require-space", action="append", default=[])
    finalize_modality.add_argument("--support-root", action="append", type=Path, default=[])
    finalize_modality.set_defaults(handler=command_finalize_modality)

    model_ready_subject = subparsers.add_parser("model-ready-subject")
    model_ready_subject.add_argument("--dataset-id", required=True)
    model_ready_subject.add_argument("--subject", required=True)
    model_ready_subject.add_argument("--finalization-markers", nargs="+", type=Path, required=True)
    model_ready_subject.add_argument("--qc-report", type=Path, required=True)
    model_ready_subject.add_argument("--output", type=Path, required=True)
    model_ready_subject.set_defaults(handler=command_model_ready_subject)

    model_ready_index = subparsers.add_parser("model-ready-index")
    model_ready_index.add_argument("--root", type=Path, required=True)
    model_ready_index.add_argument("--output-jsonl", type=Path, required=True)
    model_ready_index.add_argument("--output-tsv", type=Path, required=True)
    model_ready_index.set_defaults(handler=command_model_ready_index)

    autoconfigure = subparsers.add_parser("autoconfigure")
    autoconfigure.add_argument("--config", type=Path, required=True)
    autoconfigure.add_argument("--dataset-id")
    autoconfigure.add_argument("--bids-view", action="store_true")
    autoconfigure.add_argument("--output", type=Path, required=True)
    autoconfigure.add_argument("--minimum-confidence", type=float, default=0.90)
    autoconfigure.set_defaults(handler=command_autoconfigure)

    spatial = subparsers.add_parser("preprocess-spatial")
    spatial.add_argument("--bids-root", type=Path, required=True)
    spatial.add_argument("--output-root", type=Path, required=True)
    spatial.add_argument("--work-root", type=Path, required=True)
    spatial.add_argument("--subject", required=True)
    spatial.add_argument(
        "--modalities", nargs="+", choices=("t2w", "flair", "pd", "mra", "ct"), required=True
    )
    spatial.add_argument("--reference-root", type=Path)
    spatial.add_argument("--template", type=Path)
    spatial.add_argument("--execute", action="store_true")
    spatial.set_defaults(handler=command_preprocess_spatial)

    sensor = subparsers.add_parser("preprocess-sensor")
    sensor.add_argument("--bids-root", type=Path, required=True)
    sensor.add_argument("--output-root", type=Path, required=True)
    sensor.add_argument("--subject", required=True)
    sensor.add_argument("--session")
    sensor.add_argument("--modality", choices=("eeg", "meg", "ieeg"), required=True)
    sensor.add_argument("--low-frequency", type=float)
    sensor.add_argument("--high-frequency", type=float)
    sensor.add_argument("--resample-hz", type=float)
    sensor.add_argument("--no-ica", action="store_true")
    sensor.add_argument("--execute", action="store_true")
    sensor.set_defaults(handler=command_preprocess_sensor)

    susceptibility = subparsers.add_parser("preprocess-susceptibility")
    susceptibility.add_argument("--bids-root", type=Path, required=True)
    susceptibility.add_argument("--output-root", type=Path, required=True)
    susceptibility.add_argument("--work-root", type=Path, required=True)
    susceptibility.add_argument("--subject", required=True)
    susceptibility.add_argument("--reference-root", type=Path)
    susceptibility.add_argument("--template", type=Path)
    susceptibility.add_argument("--execute", action="store_true")
    susceptibility.set_defaults(handler=command_preprocess_susceptibility)

    assemble = subparsers.add_parser("assemble-visits")
    assemble.add_argument("--sample-index", type=Path, required=True)
    assemble.add_argument("--output-root", type=Path, required=True)
    assemble.add_argument("--expected-modalities", nargs="+", default=list(DEFAULT_MODALITIES))
    assemble.add_argument("--tolerance-days", type=int, default=30)
    assemble.add_argument("--bids-root", type=Path)
    assemble.add_argument("--dataset-id")
    assemble.set_defaults(handler=command_assemble_visits)

    grid = subparsers.add_parser("validate-grid")
    grid.add_argument("--reference", type=Path, required=True)
    grid.add_argument("--images", nargs="+", type=Path, required=True)
    grid.add_argument("--affine-tolerance", type=float, default=1e-5)
    grid.add_argument("--output", type=Path)
    grid.set_defaults(handler=command_validate_grid)

    grid_reference = subparsers.add_parser("grid-reference")
    grid_reference.add_argument("--sample-index", type=Path, required=True)
    grid_reference.add_argument("--output", type=Path, required=True)
    grid_reference.add_argument(
        "--preferred-modalities", nargs="+", default=["t1w", "t2w", "flair", "ct"]
    )
    grid_reference.add_argument("--required-space", default="MNI152NLin2009cAsym")
    grid_reference.set_defaults(handler=command_grid_reference)

    align = subparsers.add_parser("align-visit")
    align.add_argument("--visit-manifest", type=Path, required=True)
    align.add_argument(
        "--reference", type=Path, help="Exact voxel grid; omit only for sensor-only visits"
    )
    align.add_argument("--output-root", type=Path, required=True)
    align.add_argument("--sensor-transform-root", type=Path)
    align.add_argument("--require-standard-sensor-coordinates", action="store_true")
    align.set_defaults(handler=command_align_visit)

    align_all = subparsers.add_parser("align-dataset")
    align_all.add_argument("--visit-index", type=Path, required=True)
    align_all.add_argument(
        "--reference", type=Path, help="Exact voxel grid; omit only for sensor-only datasets"
    )
    align_all.add_argument("--output-root", type=Path, required=True)
    align_all.add_argument("--output-index", type=Path, required=True)
    align_all.add_argument("--sensor-transform-root", type=Path)
    align_all.add_argument("--require-standard-sensor-coordinates", action="store_true")
    align_all.set_defaults(handler=command_align_dataset)

    tensor = subparsers.add_parser("tensorize-visit")
    tensor.add_argument("--aligned-manifest", type=Path, required=True)
    tensor.add_argument("--output-root", type=Path, required=True)
    tensor.add_argument("--target-shape", nargs=3, type=int, default=(96, 112, 96))
    tensor.add_argument(
        "--image-modalities", nargs="+", default=list(TensorPolicy().image_modalities)
    )
    tensor.add_argument("--fmri-window", type=int, default=100)
    tensor.add_argument("--fmri-stride", type=int, default=50)
    tensor.add_argument("--sensor-window-seconds", type=float, default=10.0)
    tensor.add_argument("--sensor-stride-seconds", type=float, default=5.0)
    tensor.add_argument("--dwi-shells", nargs="+", type=int, default=[0, 1000, 2000, 3000])
    tensor.set_defaults(handler=command_tensorize_visit)

    tensor_all = subparsers.add_parser("tensorize-dataset")
    tensor_all.add_argument("--aligned-index", type=Path, required=True)
    tensor_all.add_argument("--output-root", type=Path, required=True)
    tensor_all.add_argument("--output-index", type=Path, required=True)
    tensor_all.add_argument("--target-shape", nargs=3, type=int, default=(96, 112, 96))
    tensor_all.add_argument(
        "--image-modalities", nargs="+", default=list(TensorPolicy().image_modalities)
    )
    tensor_all.add_argument("--fmri-window", type=int, default=100)
    tensor_all.add_argument("--fmri-stride", type=int, default=50)
    tensor_all.add_argument("--sensor-window-seconds", type=float, default=10.0)
    tensor_all.add_argument("--sensor-stride-seconds", type=float, default=5.0)
    tensor_all.add_argument("--dwi-shells", nargs="+", type=int, default=[0, 1000, 2000, 3000])
    tensor_all.set_defaults(handler=command_tensorize_dataset)

    tensor_index = subparsers.add_parser("tensor-index")
    tensor_index.add_argument("--root", type=Path, required=True)
    tensor_index.add_argument("--output", type=Path, required=True)
    tensor_index.set_defaults(handler=command_tensor_index)

    strong_qc = subparsers.add_parser("production-qc")
    strong_qc.add_argument("--finalization-markers", nargs="+", type=Path, required=True)
    strong_qc.add_argument("--thresholds", type=Path)
    strong_qc.add_argument(
        "--review-policy",
        choices=("block", "advisory"),
        default=os.environ.get("NEUROPREP_QC_REVIEW_POLICY", "block"),
    )
    strong_qc.add_argument("--output", type=Path, required=True)
    strong_qc.set_defaults(handler=command_production_qc)

    phenotype = subparsers.add_parser("phenotype-join")
    phenotype.add_argument("--visit-index", type=Path, required=True)
    phenotype.add_argument("--phenotype-table", type=Path, required=True)
    phenotype.add_argument("--output", type=Path, required=True)
    phenotype.add_argument("--report", type=Path, required=True)
    phenotype.add_argument("--participant-column", default="participant_id")
    phenotype.add_argument("--session-column", default="session_id")
    phenotype.add_argument("--date-column")
    phenotype.add_argument("--label-columns", nargs="*", default=[])
    phenotype.add_argument("--covariate-columns", nargs="*", default=[])
    phenotype.add_argument("--tolerance-days", type=int, default=60)
    phenotype.set_defaults(handler=command_phenotype_join)

    splits = subparsers.add_parser("make-splits")
    splits.add_argument("--joined-index", type=Path, required=True)
    splits.add_argument("--output", type=Path, required=True)
    splits.add_argument("--report", type=Path, required=True)
    splits.add_argument("--folds", type=int, default=5)
    splits.add_argument("--seed", type=int, default=42)
    splits.add_argument("--site-column", default="site")
    splits.add_argument("--label-column")
    splits.add_argument(
        "--strategy", choices=("stratified-group", "leave-one-site-out"), default="stratified-group"
    )
    splits.set_defaults(handler=command_make_splits)

    fold_harmonize = subparsers.add_parser("harmonize-folds")
    fold_harmonize.add_argument("--input", type=Path, required=True)
    fold_harmonize.add_argument("--output", type=Path, required=True)
    fold_harmonize.add_argument("--output-root", type=Path, required=True)
    fold_harmonize.add_argument("--report", type=Path, required=True)
    fold_harmonize.add_argument("--participant-column", default="participant_id")
    fold_harmonize.add_argument("--fold-column", default="fold")
    fold_harmonize.add_argument("--split-column", default="split")
    fold_harmonize.add_argument("--batch-column", default="site")
    fold_harmonize.add_argument("--protected-columns", nargs="*", default=[])
    fold_harmonize.add_argument("--feature-columns", nargs="+", required=True)
    fold_harmonize.add_argument("--train-label", default="train")
    fold_harmonize.set_defaults(handler=command_harmonize_folds)

    container_check = subparsers.add_parser("verify-containers")
    container_check.add_argument("--lock", type=Path, required=True)
    container_check.add_argument("--output", type=Path)
    container_check.set_defaults(handler=command_verify_containers)

    forecast = subparsers.add_parser("forecast")
    forecast.add_argument("--plan", type=Path, required=True)
    forecast.add_argument("--output", type=Path)
    forecast.set_defaults(handler=command_forecast)

    monitor = subparsers.add_parser("monitor")
    monitor.add_argument("--database", type=Path, required=True)
    monitor.add_argument("--output", type=Path)
    monitor.set_defaults(handler=command_monitor)

    exclusions = subparsers.add_parser("exclusions")
    exclusions.add_argument("--root", type=Path, required=True)
    exclusions.add_argument("--output-tsv", type=Path, required=True)
    exclusions.add_argument("--output-json", type=Path, required=True)
    exclusions.set_defaults(handler=command_exclusions)

    load_test = subparsers.add_parser("load-test")
    load_test.add_argument("--input", type=Path, required=True)
    load_test.add_argument("--repetitions", type=int, default=100)
    load_test.add_argument("--minimum-rows-per-second", type=float, default=10000.0)
    load_test.add_argument("--output", type=Path)
    load_test.set_defaults(handler=command_load_test)

    audit_production = subparsers.add_parser("production-audit")
    audit_production.add_argument("--plan", type=Path, required=True)
    audit_production.add_argument("--container-lock", type=Path, required=True)
    audit_production.add_argument("--validation-root", type=Path, required=True)
    audit_production.add_argument("--output", type=Path, required=True)
    audit_production.add_argument(
        "--required-modalities", nargs="+", default=list(DEFAULT_MODALITIES)
    )
    audit_production.set_defaults(handler=command_production_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
