from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import nibabel as nib
import numpy as np

from .experts import (
    ExpertDecision,
    ExpertRegistry,
    OUTPUT_CONTRACT_VERSION,
    VisitContext,
)
from .grouping import select_primary_reference
from .ingest import ingest_command, read_ingest_manifest
from .models import DatasetConfig, ImageRecord, Inventory, Modality, Plan, Space, Task, TaskStage
from .provenance import task_fingerprint
from .profiles import execution_profile, fmri_task_labels, is_resting_fmri
from .references import select_reference
from .registry import PipelineRegistry


MODALITY_PIPELINES = {
    Modality.FMRI: "fmriprep",
    Modality.DWI: "qsiprep",
    Modality.PET: "petprep",
    Modality.ASL: "aslprep",
    Modality.SWI: "neuroprep-susceptibility",
    Modality.CT: "neuroprep-ct",
    Modality.MRA: "neuroprep-mra",
    Modality.EEG: "mne-bids-pipeline-eeg",
    Modality.MEG: "mne-bids-pipeline-meg",
    Modality.IEEG: "mne-bids-pipeline-ieeg",
}

STRUCTURAL_MODALITIES = {Modality.T1W, Modality.T2W, Modality.FLAIR, Modality.PD}
SPATIAL_MODALITIES = STRUCTURAL_MODALITIES | {
    Modality.FMRI,
    Modality.DWI,
    Modality.PET,
    Modality.ASL,
    Modality.SWI,
    Modality.CT,
    Modality.MRA,
}
ELECTROPHYS_MODALITIES = {Modality.EEG, Modality.MEG, Modality.IEEG}


def _safe_id(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "-" for character in value
    )


def build_plan(
    dataset: DatasetConfig,
    inventory: Inventory,
    registry: PipelineRegistry | None = None,
) -> Plan:
    registry = registry or PipelineRegistry.default()
    tasks: list[Task] = []
    warnings = list(inventory.warnings)

    if dataset.input_layout != "bids":
        _add_ingest_plan(dataset, registry, tasks, warnings)
        _finalize_tasks(tasks)
        return Plan(
            version="3.0",
            dataset_id=dataset.dataset_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            tasks=tasks,
            warnings=sorted(set(warnings)),
        )

    validate_id = _add_validation_task(dataset, registry, tasks, [])
    subject_release_ids = _add_subject_tasks(
        dataset, inventory, registry, validate_id, tasks, warnings
    )
    if subject_release_ids:
        index_id = _add_dataset_model_ready_index_task(
            dataset, registry, subject_release_ids, tasks
        )
        if bool(dataset.metadata.get("build_training_tensors", True)):
            _add_dataset_training_tasks(
                dataset,
                registry,
                index_id,
                tasks,
                warnings,
                has_spatial_modalities=bool(set(inventory.modalities()) & SPATIAL_MODALITIES),
            )
    _validate_dependencies(tasks)
    _finalize_tasks(tasks)
    return Plan(
        version="3.0",
        dataset_id=dataset.dataset_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        tasks=tasks,
        warnings=sorted(set(warnings)),
    )


def _add_ingest_plan(
    dataset: DatasetConfig,
    registry: PipelineRegistry,
    tasks: list[Task],
    warnings: list[str],
) -> None:
    ingest_ids: list[str] = []
    manifest_value = dataset.metadata.get("ingest_manifest")
    if manifest_value:
        records = read_ingest_manifest(Path(str(manifest_value)).expanduser())
        for record in records:
            identity = record.participant
            if record.session:
                identity += f"--{record.session}"
            task_id = f"{dataset.dataset_id}--ingest--{_safe_id(identity)}"
            ingest_ids.append(task_id)
            tasks.append(
                Task(
                    task_id=task_id,
                    dataset_id=dataset.dataset_id,
                    stage=TaskStage.INGEST,
                    pipeline="dcm2bids",
                    subject=record.participant,
                    session=record.session,
                    inputs=[
                        str(value) for value in (record.source, record.config) if value is not None
                    ],
                    outputs=[
                        str(
                            dataset.bids_root
                            / f"sub-{record.participant}"
                            / (f"ses-{record.session}" if record.session else "")
                        )
                    ],
                    params={
                        "converter": record.converter,
                        "conversion_is_reviewed": True,
                        "output_contract_version": OUTPUT_CONTRACT_VERSION,
                    },
                    resources={"cpus": 4, "memory_gb": 8, "time": "12:00:00"},
                    command=ingest_command(record, dataset),
                )
            )
    else:
        task_id = f"{dataset.dataset_id}--ingest"
        ingest_ids.append(task_id)
        tasks.append(
            Task(
                task_id=task_id,
                dataset_id=dataset.dataset_id,
                stage=TaskStage.INGEST,
                pipeline="dcm2bids",
                inputs=[str(dataset.root)],
                outputs=[str(dataset.bids_root)],
                params={
                    "requires_ingest_manifest": True,
                    "output_contract_version": OUTPUT_CONTRACT_VERSION,
                },
                resources={"cpus": 4, "memory_gb": 8, "time": "12:00:00"},
            )
        )
        warnings.append(
            "Non-BIDS input is blocked until metadata.ingest_manifest provides reviewed "
            "participant/session mappings"
        )
    _add_validation_task(dataset, registry, tasks, ingest_ids)
    warnings.append(
        "This is an ingest-only plan. After conversion, inventory the BIDS root and build a "
        "separate preprocessing plan."
    )


def _add_validation_task(
    dataset: DatasetConfig,
    registry: PipelineRegistry,
    tasks: list[Task],
    dependencies: list[str],
) -> str:
    task_id = f"{dataset.dataset_id}--validate"
    output = dataset.derivatives_root / "validation" / "bids-validator.json"
    validator_command = registry.command("bids-validator", dataset)
    command = (
        [
            sys.executable,
            "-m",
            "neuroprep.cli",
            "capture",
            "--output",
            str(output),
            "--",
            *validator_command,
        ]
        if validator_command
        else []
    )
    tasks.append(
        Task(
            task_id=task_id,
            dataset_id=dataset.dataset_id,
            stage=TaskStage.VALIDATE,
            pipeline="bids-validator",
            dependencies=dependencies,
            inputs=[str(dataset.bids_root)],
            outputs=[str(output)],
            params={
                "output_contract_version": OUTPUT_CONTRACT_VERSION,
                "verify_outputs": True,
            },
            resources=registry.resources("bids-validator"),
            command=command,
        )
    )
    return task_id


def _add_subject_tasks(
    dataset: DatasetConfig,
    inventory: Inventory,
    registry: PipelineRegistry,
    validate_id: str,
    tasks: list[Task],
    warnings: list[str],
) -> list[str]:
    include_standard_space = bool(dataset.metadata.get("run_standard_space", True))
    profile = execution_profile(dataset)
    expert_registry = ExpertRegistry()
    by_subject: dict[str, list[ImageRecord]] = defaultdict(list)
    for image in inventory.images:
        if image.subject != "__unknown__" and not image.is_derivative:
            by_subject[image.subject].append(image)

    subject_release_ids: list[str] = []

    for subject in sorted(by_subject):
        images = by_subject[subject]
        subject_modalities = {image.modality for image in images}
        subject_token = _safe_id(subject)
        preprocessing_ids: list[str] = []
        postprocessing_ids: list[str] = []
        pipeline_task_ids: dict[str, str] = {}
        finalization_markers: list[Path] = []
        all_visit_decisions = expert_registry.visit_decisions(
            images, dataset.metadata, include_missing=True
        )
        visit_contexts = expert_registry.visit_contexts(
            images,
            tolerance_days=int(dataset.metadata.get("visit_tolerance_days", 30)),
        )
        decisions = [decision for decision in all_visit_decisions if decision.present]
        visit_policy = _visit_policy_payload(all_visit_decisions)
        target_spaces = [Space.NATIVE.value, Space.SUBJECT.value]
        if include_standard_space:
            target_spaces.append(Space.STANDARD.value)

        has_t1w = Modality.T1W in subject_modalities
        has_fmri = Modality.FMRI in subject_modalities
        eligible_fmri_visits = [
            context
            for context in visit_contexts
            if {Modality.T1W, Modality.FMRI} <= {image.modality for image in context.images}
        ]
        pipelines: set[str] = set()
        if has_t1w:
            pipelines.add(profile.t1w_pipeline)
        if subject_modalities & {Modality.T2W, Modality.FLAIR, Modality.PD}:
            pipelines.add("neuroprep-structural")
        for modality, pipeline in MODALITY_PIPELINES.items():
            if modality is Modality.FMRI:
                continue
            if modality in subject_modalities:
                pipelines.add(pipeline)

        fmri_preprocess_pipeline: str | None = None
        if has_fmri:
            if not has_t1w:
                warnings.append(
                    f"fMRI for subject {subject} is not scheduled because neither "
                    "DeepPrep nor fMRIPrep can run without a structural reference; "
                    "the raw visit remains in the assembly inventory as acquired but unreleased"
                )
            elif not eligible_fmri_visits:
                warnings.append(
                    f"fMRI for subject {subject} is not scheduled because no visit contains "
                    "both fMRI and a visit-local T1w; cross-visit anatomical borrowing is "
                    "forbidden and affected visits remain acquired but unreleased"
                )
            elif profile.fmri_pipeline == "deepprep":
                fmri_preprocess_pipeline = "deepprep"
                pipelines.add("deepprep")
            else:
                fmri_preprocess_pipeline = "fmriprep"
                pipelines.add("fmriprep")

        anatomical_pipeline = profile.t1w_pipeline if has_t1w else None
        structural_id = (
            f"{dataset.dataset_id}--{subject_token}--{anatomical_pipeline}"
            if anatomical_pipeline
            else None
        )
        deepprep_visit_tasks: dict[str, tuple[str, Path]] = {}

        for pipeline in sorted(pipelines):
            pipeline_decisions = _decisions_for_pipeline(pipeline, decisions)
            decision = _aggregate_pipeline_decision(pipeline, pipeline_decisions)
            relevant = _pipeline_inputs(pipeline, images)
            acquisition_policy: dict[str, Any] | None = None
            task_id = f"{dataset.dataset_id}--{subject_token}--{pipeline}"
            if pipeline == "deepprep":
                task_id, deepprep_visit_tasks = _add_visit_scoped_deepprep_tasks(
                    dataset=dataset,
                    registry=registry,
                    validate_id=validate_id,
                    subject=subject,
                    subject_token=subject_token,
                    contexts=visit_contexts,
                    pipeline_decisions=pipeline_decisions,
                    visit_policy=visit_policy,
                    include_standard_space=include_standard_space,
                    tasks=tasks,
                    warnings=warnings,
                )
                preprocessing_ids.append(task_id)
                pipeline_task_ids[pipeline] = task_id
                continue
            preprocessing_ids.append(task_id)
            pipeline_task_ids[pipeline] = task_id
            command = registry.command(pipeline, dataset, subject)
            native_command = _native_upstream_command(
                pipeline,
                dataset,
                subject,
                relevant_modalities={image.modality for image in images},
                anatomical_pipeline=anatomical_pipeline,
            )
            if native_command:
                command = native_command
            if pipeline == "smriprep":
                command = _configure_smriprep_spaces(
                    command,
                    dataset,
                    include_standard_space,
                )
            elif pipeline == "deepprep":
                command = _configure_deepprep(
                    command,
                    dataset,
                    include_standard_space=include_standard_space,
                    has_fmri=has_fmri,
                    task_labels=fmri_task_labels(images),
                )
            elif profile.name == "high-throughput" and pipeline == "qsiprep":
                acquisition_policy = _qsiprep_acquisition_policy(relevant)
                command = _configure_qsiprep_dwi_only(
                    command,
                    separate_all_dwis=bool(acquisition_policy["separate_all_dwis"]),
                )
            elif profile.name == "high-throughput" and pipeline in {"petprep", "aslprep"}:
                command = _remove_cli_option(command, "--derivatives", values=1)
            if decision and decision.status == "blocked":
                command = []
                warnings.append(
                    f"{pipeline} for subject {subject} blocked by: {', '.join(decision.blockers)}"
                )
            if not command:
                spec = registry.spec(pipeline)
                warnings.append(
                    f"{pipeline} for subject {subject} needs a reviewed command: "
                    f"{spec.get('note', 'no command template registered')}"
                )
            pipeline_root = _pipeline_derivative_root(dataset, pipeline)
            tasks.append(
                Task(
                    task_id=task_id,
                    dataset_id=dataset.dataset_id,
                    stage=TaskStage.PREPROCESS,
                    pipeline=pipeline,
                    subject=subject,
                    dependencies=[
                        validate_id,
                        *(
                            [structural_id]
                            if structural_id
                            and pipeline != anatomical_pipeline
                            and (
                                (
                                    profile.reuse_external_anatomy
                                    and pipeline in {"fmriprep", "qsiprep", "petprep", "aslprep"}
                                )
                                or pipeline
                                in {
                                    "neuroprep-structural",
                                    "neuroprep-susceptibility",
                                    "neuroprep-mra",
                                    "neuroprep-ct",
                                }
                            )
                            else []
                        ),
                    ],
                    inputs=[str(image.path) for image in relevant],
                    outputs=[
                        str(_preprocess_subject_output(pipeline_root, pipeline, subject_token))
                    ],
                    params={
                        "modalities": sorted({image.modality.value for image in relevant}),
                        "sessions": sorted({image.session for image in relevant if image.session}),
                        "expert_decision": decision.to_dict() if decision else None,
                        "specialist_decisions": [item.to_dict() for item in pipeline_decisions],
                        "expert": decision.expert if decision else "custom",
                        "output_kind": decision.output_kind if decision else "spatial-image",
                        "visit_policy": visit_policy,
                        "execution_profile": profile.to_dict(),
                        "bids_validation_dependency": validate_id,
                        "output_contract_version": OUTPUT_CONTRACT_VERSION,
                        "verify_outputs": bool(dataset.metadata.get("verify_outputs", True)),
                        **(
                            {"acquisition_policy": acquisition_policy}
                            if acquisition_policy is not None
                            else {}
                        ),
                    },
                    resources=_resources_for(pipeline, dataset, registry),
                    command=command,
                )
            )

        if structural_id:
            final_id, marker = _add_anatomical_finalization_task(
                dataset,
                registry,
                subject,
                subject_token,
                structural_id,
                anatomical_pipeline or "smriprep",
                include_standard_space,
                tasks,
                source_root_override=(
                    _pipeline_derivative_root(dataset, "deepprep") if deepprep_visit_tasks else None
                ),
            )
            postprocessing_ids.append(final_id)
            finalization_markers.append(marker)

        fmri_release_pipeline: str | None = None
        if fmri_preprocess_pipeline == "fmriprep":
            fmriprep_id = f"{dataset.dataset_id}--{subject_token}--fmriprep"
            xcpd_id = f"{dataset.dataset_id}--{subject_token}--xcpd"
            pipeline_task_ids["xcpd"] = xcpd_id
            run_xcpd = bool(dataset.metadata.get("run_xcpd", True))
            xcpd_command = registry.command("xcpd", dataset, subject) if run_xcpd else []
            if not run_xcpd:
                warnings.append(
                    f"xcpd for subject {subject} is required for the model-ready fMRI tier; "
                    "run_xcpd=false blocks release"
                )
            tasks.append(
                Task(
                    task_id=xcpd_id,
                    dataset_id=dataset.dataset_id,
                    stage=TaskStage.DENOISE,
                    pipeline="xcpd",
                    subject=subject,
                    dependencies=[fmriprep_id],
                    outputs=[str(dataset.derivatives_root / "xcpd" / f"sub-{subject_token}")],
                    params={
                        "modalities": [Modality.FMRI.value],
                        "output_kind": "denoised-spatiotemporal-image",
                        "output_contract_version": OUTPUT_CONTRACT_VERSION,
                        "verify_outputs": True,
                    },
                    resources=_resources_for("xcpd", dataset, registry),
                    command=xcpd_command,
                )
            )
            fmri_release_pipeline = "xcpd"
        elif fmri_preprocess_pipeline == "deepprep":
            deepprep_id = f"{dataset.dataset_id}--{subject_token}--deepprep"
            run_postprocess = bool(dataset.metadata.get("run_deepprep_postprocess", True))
            resting = is_resting_fmri(images, dataset)
            if resting and run_postprocess:
                pipeline = "deepprep-postprocess"
                if deepprep_visit_tasks:
                    postprocess_id = _add_visit_scoped_deepprep_postprocess_tasks(
                        dataset=dataset,
                        registry=registry,
                        subject=subject,
                        subject_token=subject_token,
                        contexts=visit_contexts,
                        deepprep_visit_tasks=deepprep_visit_tasks,
                        include_standard_space=include_standard_space,
                        tasks=tasks,
                    )
                else:
                    postprocess_id = f"{dataset.dataset_id}--{subject_token}--{pipeline}"
                command = (
                    _configure_deepprep_postprocess(
                        registry.command(pipeline, dataset, subject),
                        dataset,
                        fmri_task_labels(images),
                        include_standard_space,
                    )
                    if not deepprep_visit_tasks
                    else []
                )
                if postprocess_id is None:
                    warnings.append(
                        f"No visit-local fMRI output is eligible for DeepPrep "
                        f"postprocessing for subject {subject}"
                    )
                    fmri_release_pipeline = "deepprep"
                    postprocess_id = ""
                pipeline_task_ids[pipeline] = postprocess_id
                if not deepprep_visit_tasks:
                    tasks.append(
                        Task(
                            task_id=postprocess_id,
                            dataset_id=dataset.dataset_id,
                            stage=TaskStage.DENOISE,
                            pipeline=pipeline,
                            subject=subject,
                            dependencies=[deepprep_id],
                            inputs=[
                                str(
                                    _anatomical_source_root(dataset, "deepprep", subject_token)
                                    / f"sub-{subject_token}"
                                )
                            ],
                            outputs=[
                                str(dataset.derivatives_root / pipeline / f"sub-{subject_token}")
                            ],
                            params={
                                "modalities": [Modality.FMRI.value],
                                "analysis": "resting-state",
                                "output_kind": "denoised-spatiotemporal-image",
                                "execution_profile": profile.to_dict(),
                                "output_contract_version": OUTPUT_CONTRACT_VERSION,
                                "verify_outputs": True,
                            },
                            resources=_resources_for(pipeline, dataset, registry),
                            command=command,
                        )
                    )
                if postprocess_id:
                    fmri_release_pipeline = pipeline
            else:
                fmri_release_pipeline = "deepprep"
                reason = (
                    "the acquisition is task fMRI"
                    if not resting
                    else "run_deepprep_postprocess=false"
                )
                warnings.append(
                    f"DeepPrep resting-state postprocessing was not applied to subject {subject}: "
                    f"{reason}. The release contains motion/distortion-corrected BOLD plus "
                    "confounds and preserves task signal."
                )

        for modality in sorted(
            subject_modalities - {Modality.T1W, Modality.FIELDMAP}, key=lambda value: value.value
        ):
            if modality in {Modality.T2W, Modality.FLAIR, Modality.PD}:
                pipeline = "neuroprep-structural"
            elif modality is Modality.FMRI:
                pipeline = fmri_release_pipeline
            else:
                pipeline = MODALITY_PIPELINES.get(modality)
            if pipeline is None or pipeline not in pipeline_task_ids:
                continue
            final_id, marker = _add_modality_finalization_task(
                dataset,
                registry,
                subject,
                subject_token,
                modality,
                pipeline,
                pipeline_task_ids[pipeline],
                include_standard_space,
                tasks,
                pipeline_root_override=(
                    _pipeline_derivative_root(dataset, pipeline)
                    if deepprep_visit_tasks and pipeline in {"deepprep", "deepprep-postprocess"}
                    else None
                ),
            )
            postprocessing_ids.append(final_id)
            finalization_markers.append(marker)

        registration_id = _add_registration_task(
            dataset,
            registry,
            subject,
            subject_token,
            images,
            subject_modalities,
            preprocessing_ids,
            target_spaces,
            include_standard_space,
            tasks,
            warnings,
        )
        qc_dependencies = (
            [registration_id, *postprocessing_ids]
            if registration_id
            else [*preprocessing_ids, *postprocessing_ids]
        )
        qc_id = _add_qc_task(
            dataset,
            registry,
            subject,
            subject_token,
            images,
            registration_id,
            qc_dependencies or [validate_id],
            finalization_markers,
            tasks,
        )
        subject_release_ids.append(
            _add_model_ready_task(
                dataset,
                registry,
                subject,
                subject_token,
                finalization_markers,
                qc_id,
                tasks,
            )
        )
    return subject_release_ids


def _decisions_for_pipeline(
    pipeline: str,
    decisions: list[ExpertDecision],
) -> list[ExpertDecision]:
    modalities_by_pipeline: dict[str, set[str]] = {
        "smriprep": {Modality.T1W.value},
        "deepprep": {Modality.T1W.value, Modality.FMRI.value},
        "neuroprep-structural": {
            Modality.T2W.value,
            Modality.FLAIR.value,
            Modality.PD.value,
        },
        "fmriprep": {Modality.FMRI.value},
        "qsiprep": {Modality.DWI.value},
        "petprep": {Modality.PET.value},
        "aslprep": {Modality.ASL.value},
        "neuroprep-susceptibility": {Modality.SWI.value},
        "neuroprep-mra": {Modality.MRA.value},
        "neuroprep-ct": {Modality.CT.value},
        "mne-bids-pipeline-eeg": {Modality.EEG.value},
        "mne-bids-pipeline-meg": {Modality.MEG.value},
        "mne-bids-pipeline-ieeg": {Modality.IEEG.value},
    }
    expected = modalities_by_pipeline.get(pipeline, set())
    return [decision for decision in decisions if set(decision.modalities) & expected]


def _add_visit_scoped_deepprep_tasks(
    dataset: DatasetConfig,
    registry: PipelineRegistry,
    validate_id: str,
    subject: str,
    subject_token: str,
    contexts: list[VisitContext],
    pipeline_decisions: list[ExpertDecision],
    visit_policy: dict[str, Any],
    include_standard_space: bool,
    tasks: list[Task],
    warnings: list[str],
) -> tuple[str, dict[str, tuple[str, Path]]]:
    """Run DeepPrep independently for every visit.

    DeepPrep has a participant selector but no session selector.  Feeding all
    sessions at once makes FreeSurfer combine visits as repeat T1 runs and can
    fail when voxel geometry differs. A visit may also contain alternate T1w
    protocols that are inappropriate for structural averaging. NeuroPrep
    therefore creates a hard-linked view with one ranked primary T1w and
    isolates output and work state for every visit.
    """

    base_id = f"{dataset.dataset_id}--{subject_token}--deepprep"
    output_root = _pipeline_derivative_root(dataset, "deepprep")
    tolerance_days = int(dataset.metadata.get("visit_tolerance_days", 30))
    visit_tasks: dict[str, tuple[str, Path]] = {}
    for context in contexts:
        modalities = {image.modality for image in context.images}
        if Modality.T1W not in modalities:
            if Modality.FMRI in modalities:
                warnings.append(
                    f"DeepPrep visit {context.visit_id} for subject {subject} has fMRI "
                    "but no visit-local T1w; it is fail-closed rather than borrowing "
                    "anatomy across visits, and remains acquired-but-unreleased in assembly"
                )
            continue
        visit_token = _safe_id(context.visit_id)
        view_root = (
            dataset.work_root / "visit-views" / f"sub-{subject_token}" / visit_token / "bids"
        )
        stage_marker = view_root.parent / "staging.json"
        t1w_images = [image for image in context.images if image.modality is Modality.T1W]
        primary_t1 = select_primary_reference(t1w_images, dataset) or select_reference(t1w_images)
        stage_id = f"{dataset.dataset_id}--{subject_token}--{visit_token}--stage-visit"
        tasks.append(
            Task(
                task_id=stage_id,
                dataset_id=dataset.dataset_id,
                stage=TaskStage.INGEST,
                pipeline="neuroprep-stage-visit",
                subject=subject,
                session=context.visit_id,
                dependencies=[validate_id],
                inputs=[str(image.path) for image in context.images],
                outputs=[str(stage_marker)],
                params={
                    "visit_context": context.to_dict(),
                    "link_mode": "hardlink",
                    "output_contract_version": OUTPUT_CONTRACT_VERSION,
                    "verify_outputs": True,
                },
                resources={"cpus": 1, "memory_gb": 2, "time": "01:00:00"},
                command=[
                    sys.executable,
                    "-m",
                    "neuroprep.cli",
                    "stage-visit",
                    "--bids-root",
                    str(dataset.bids_root),
                    "--output",
                    str(view_root),
                    "--subject",
                    subject,
                    "--visit-id",
                    context.visit_id,
                    "--marker",
                    str(stage_marker),
                    "--tolerance-days",
                    str(tolerance_days),
                    *(["--primary-t1", str(primary_t1.path)] if primary_t1 is not None else []),
                ],
            )
        )

        visit_output = output_root / f"sub-{subject_token}" / "visits" / visit_token
        command = registry.command("deepprep", dataset, subject)
        command = _configure_deepprep(
            command,
            dataset,
            include_standard_space=include_standard_space,
            has_fmri=Modality.FMRI in modalities,
            task_labels=fmri_task_labels(list(context.images)),
        )
        command = _rewrite_deepprep_visit_command(
            command,
            dataset=dataset,
            subject_token=subject_token,
            view_root=view_root,
            visit_output=visit_output,
        )
        decisions = [
            decision for decision in pipeline_decisions if decision.visit_id == context.visit_id
        ]
        aggregate = _aggregate_pipeline_decision("deepprep", decisions)
        if aggregate and aggregate.status == "blocked":
            command = []
        task_id = f"{dataset.dataset_id}--{subject_token}--{visit_token}--deepprep"
        visit_tasks[context.visit_id] = (task_id, visit_output)
        tasks.append(
            Task(
                task_id=task_id,
                dataset_id=dataset.dataset_id,
                stage=TaskStage.PREPROCESS,
                pipeline="deepprep",
                subject=subject,
                session=context.visit_id,
                dependencies=[stage_id],
                inputs=[
                    str(image.path) for image in _pipeline_inputs("deepprep", list(context.images))
                ],
                outputs=[
                    str(
                        visit_output
                        / ("BOLD" if Modality.FMRI in modalities else "Recon")
                        / f"sub-{subject_token}"
                    )
                ],
                params={
                    "modalities": sorted(modality.value for modality in modalities),
                    "sessions": [context.visit_id],
                    "expert_decision": aggregate.to_dict() if aggregate else None,
                    "specialist_decisions": [decision.to_dict() for decision in decisions],
                    "visit_context": context.to_dict(),
                    "visit_policy": visit_policy,
                    "execution_scope": "visit",
                    "bids_view": str(view_root),
                    "execution_profile": execution_profile(dataset).to_dict(),
                    "bids_validation_dependency": validate_id,
                    "output_contract_version": OUTPUT_CONTRACT_VERSION,
                    "verify_outputs": bool(dataset.metadata.get("verify_outputs", True)),
                },
                resources=_resources_for("deepprep", dataset, registry),
                command=command,
            )
        )

    if not visit_tasks:
        warnings.append(
            f"No longitudinal visit for subject {subject} has the T1w required by DeepPrep"
        )
    marker = output_root / f"sub-{subject_token}" / "visit-execution.json"
    tasks.append(
        Task(
            task_id=base_id,
            dataset_id=dataset.dataset_id,
            stage=TaskStage.POSTPROCESS,
            pipeline="deepprep-visit-index",
            subject=subject,
            dependencies=[task_id for task_id, _ in visit_tasks.values()],
            inputs=[str(root) for _, root in visit_tasks.values()],
            outputs=[str(marker)],
            params={
                "specialist_decisions": [decision.to_dict() for decision in pipeline_decisions],
                "visit_policy": visit_policy,
                "execution_scope": "subject-longitudinal-barrier",
                "output_contract_version": OUTPUT_CONTRACT_VERSION,
                "verify_outputs": True,
            },
            resources={"cpus": 1, "memory_gb": 2, "time": "00:30:00"},
            command=(
                [
                    sys.executable,
                    "-m",
                    "neuroprep.cli",
                    "visit-execution-index",
                    "--roots",
                    *[str(root) for _, root in visit_tasks.values()],
                    "--marker",
                    str(marker),
                    "--subject",
                    subject,
                    "--pipeline",
                    "deepprep",
                ]
                if visit_tasks
                else []
            ),
        )
    )
    return base_id, visit_tasks


def _rewrite_deepprep_visit_command(
    command: list[str],
    dataset: DatasetConfig,
    subject_token: str,
    view_root: Path,
    visit_output: Path,
) -> list[str]:
    source = str(dataset.bids_root)
    configured = [
        value.replace(source, str(view_root)) if source in value else value for value in command
    ]
    container_output = f"/out/deepprep/sub-{subject_token}"
    visit_container_output = f"/out/deepprep/sub-{subject_token}/visits/{visit_output.name}"
    return [visit_container_output if value == container_output else value for value in configured]


def _add_visit_scoped_deepprep_postprocess_tasks(
    dataset: DatasetConfig,
    registry: PipelineRegistry,
    subject: str,
    subject_token: str,
    contexts: list[VisitContext],
    deepprep_visit_tasks: dict[str, tuple[str, Path]],
    include_standard_space: bool,
    tasks: list[Task],
) -> str | None:
    base_id = f"{dataset.dataset_id}--{subject_token}--deepprep-postprocess"
    output_root = _pipeline_derivative_root(dataset, "deepprep-postprocess")
    visit_task_ids: list[str] = []
    visit_outputs: list[Path] = []
    for context in contexts:
        if context.visit_id not in deepprep_visit_tasks or Modality.FMRI not in {
            image.modality for image in context.images
        }:
            continue
        deepprep_id, deepprep_output = deepprep_visit_tasks[context.visit_id]
        visit_token = _safe_id(context.visit_id)
        visit_output = output_root / f"sub-{subject_token}" / "visits" / visit_token
        command = _configure_deepprep_postprocess(
            registry.command("deepprep-postprocess", dataset, subject),
            dataset,
            fmri_task_labels(list(context.images)),
            include_standard_space,
        )
        command = [
            (
                f"/out/deepprep/{deepprep_output.relative_to(_pipeline_derivative_root(dataset, 'deepprep'))}/BOLD"
                if value == f"/out/deepprep/sub-{subject_token}/BOLD"
                else (
                    f"/out/deepprep-postprocess/sub-{subject_token}/visits/{visit_token}"
                    if value == f"/out/deepprep-postprocess/sub-{subject_token}"
                    else value
                )
            )
            for value in command
        ]
        task_id = f"{dataset.dataset_id}--{subject_token}--{visit_token}--deepprep-postprocess"
        visit_task_ids.append(task_id)
        visit_outputs.append(visit_output)
        tasks.append(
            Task(
                task_id=task_id,
                dataset_id=dataset.dataset_id,
                stage=TaskStage.DENOISE,
                pipeline="deepprep-postprocess",
                subject=subject,
                session=context.visit_id,
                dependencies=[deepprep_id],
                inputs=[str(deepprep_output / "BOLD" / f"sub-{subject_token}")],
                outputs=[str(visit_output)],
                params={
                    "modalities": [Modality.FMRI.value],
                    "analysis": "resting-state",
                    "visit_context": context.to_dict(),
                    "execution_scope": "visit",
                    "output_contract_version": OUTPUT_CONTRACT_VERSION,
                    "verify_outputs": True,
                },
                resources=_resources_for("deepprep-postprocess", dataset, registry),
                command=command,
            )
        )
    if not visit_task_ids:
        return None
    marker = output_root / f"sub-{subject_token}" / "visit-execution.json"
    tasks.append(
        Task(
            task_id=base_id,
            dataset_id=dataset.dataset_id,
            stage=TaskStage.POSTPROCESS,
            pipeline="deepprep-postprocess-visit-index",
            subject=subject,
            dependencies=visit_task_ids,
            inputs=[str(path) for path in visit_outputs],
            outputs=[str(marker)],
            params={
                "execution_scope": "subject-longitudinal-barrier",
                "output_contract_version": OUTPUT_CONTRACT_VERSION,
                "verify_outputs": True,
            },
            resources={"cpus": 1, "memory_gb": 2, "time": "00:30:00"},
            command=[
                sys.executable,
                "-m",
                "neuroprep.cli",
                "visit-execution-index",
                "--roots",
                *[str(path) for path in visit_outputs],
                "--marker",
                str(marker),
                "--subject",
                subject,
                "--pipeline",
                "deepprep-postprocess",
            ],
        )
    )
    return base_id


def _aggregate_pipeline_decision(
    pipeline: str,
    decisions: list[ExpertDecision],
) -> ExpertDecision | None:
    if not decisions:
        return None
    blockers = tuple(sorted({blocker for decision in decisions for blocker in decision.blockers}))
    warnings = tuple(sorted({warning for decision in decisions for warning in decision.warnings}))
    statuses = {decision.status for decision in decisions}
    status = "blocked" if "blocked" in statuses else "review" if warnings else "ready"
    return ExpertDecision(
        expert="+".join(sorted({decision.expert for decision in decisions})),
        pipeline=pipeline,
        status=status,
        modalities=tuple(
            sorted({modality for decision in decisions for modality in decision.modalities})
        ),
        inputs=tuple(sorted({path for decision in decisions for path in decision.inputs})),
        output_kind="+".join(sorted({decision.output_kind for decision in decisions})),
        native_space="+".join(sorted({decision.native_space for decision in decisions})),
        policy=";".join(sorted({decision.policy for decision in decisions if decision.policy}))
        or None,
        blockers=blockers,
        warnings=warnings,
        subject=decisions[0].subject,
        visit_count=max(decision.visit_count or 1 for decision in decisions),
        visit_mode=(
            "longitudinal"
            if any((decision.visit_count or 1) > 1 for decision in decisions)
            else "single-visit"
        ),
        longitudinal_strategy="coordinated-per-visit",
        missing_policy="explicit-missing-modality-mask",
    )


def _visit_policy_payload(
    decisions: list[ExpertDecision],
) -> dict[str, Any]:
    visits: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if decision.visit_id is None:
            continue
        visit = visits.setdefault(
            decision.visit_id,
            {
                "visit_id": decision.visit_id,
                "visit_index": decision.visit_index,
                "visit_count": decision.visit_count,
                "mode": decision.visit_mode,
                "present_modalities": [],
                "missing_modalities": [],
                "missing_modality_mask": {},
            },
        )
        modality = decision.modalities[0]
        target = "present_modalities" if decision.present else "missing_modalities"
        visit[target].append(modality)
        visit["missing_modality_mask"][modality] = int(decision.present)
    ordered = sorted(
        visits.values(),
        key=lambda visit: (int(visit["visit_index"] or 0), str(visit["visit_id"])),
    )
    for visit in ordered:
        visit["present_modalities"].sort()
        visit["missing_modalities"].sort()
        visit["missing_modality_mask"] = dict(sorted(visit["missing_modality_mask"].items()))
    return {
        "mode": (
            "longitudinal"
            if any(int(visit.get("visit_count") or 1) > 1 for visit in ordered)
            else "single-visit"
        ),
        "visit_count": len(ordered),
        "visits": ordered,
        "cross_visit_policy": (
            "preprocess each BIDS session independently, preserve visit identity, and "
            "use the validated common training grid for longitudinal stacking without "
            "mixing acquisitions across visits"
        ),
    }


def _pipeline_inputs(pipeline: str, images: list[ImageRecord]) -> list[ImageRecord]:
    primary: set[Modality]
    if pipeline == "deepprep":
        primary = {Modality.T1W, Modality.FMRI}
    elif pipeline in {"smriprep", "neuroprep-structural"}:
        primary = STRUCTURAL_MODALITIES
    else:
        primary = {
            modality for modality, candidate in MODALITY_PIPELINES.items() if candidate == pipeline
        }
    supporting = set()
    if pipeline in {
        "fmriprep",
        "deepprep",
        "qsiprep",
        "petprep",
        "aslprep",
        "neuroprep-susceptibility",
        "neuroprep-mra",
    }:
        supporting |= STRUCTURAL_MODALITIES
    if pipeline in {"fmriprep", "deepprep", "qsiprep", "aslprep"}:
        supporting.add(Modality.FIELDMAP)
    return [image for image in images if image.modality in primary | supporting]


def _native_upstream_command(
    pipeline: str,
    dataset: DatasetConfig,
    subject: str,
    relevant_modalities: set[Modality],
    anatomical_pipeline: str | None = None,
) -> list[str]:
    subject_token = subject.removeprefix("sub-")
    spatial_modalities: list[str] = []
    if pipeline == "neuroprep-structural":
        spatial_modalities = [
            modality.value
            for modality in (Modality.T2W, Modality.FLAIR, Modality.PD)
            if modality in relevant_modalities
        ]
    elif pipeline == "neuroprep-mra":
        spatial_modalities = ["mra"]
    elif pipeline == "neuroprep-ct":
        spatial_modalities = ["ct"]
    if spatial_modalities:
        command = [
            sys.executable,
            "-m",
            "neuroprep.cli",
            "preprocess-spatial",
            "--bids-root",
            str(dataset.bids_root),
            "--output-root",
            str(dataset.derivatives_root / pipeline),
            "--work-root",
            str(dataset.work_root / pipeline),
            "--subject",
            subject_token,
            "--modalities",
            *spatial_modalities,
            "--reference-root",
            str(_anatomical_source_root(dataset, anatomical_pipeline or "smriprep", subject_token)),
            "--execute",
        ]
        template = dataset.metadata.get("standard_template") or dataset.metadata.get("mni_template")
        if template:
            command.extend(["--template", str(template)])
        return _containerize_native_upstream(command, dataset)
    sensor_pipeline = {
        "mne-bids-pipeline-eeg": "eeg",
        "mne-bids-pipeline-meg": "meg",
        "mne-bids-pipeline-ieeg": "ieeg",
    }
    if pipeline in sensor_pipeline:
        modality = sensor_pipeline[pipeline]
        return [
            sys.executable,
            "-m",
            "neuroprep.cli",
            "preprocess-sensor",
            "--bids-root",
            str(dataset.bids_root),
            "--output-root",
            str(dataset.derivatives_root / pipeline),
            "--subject",
            subject_token,
            "--modality",
            modality,
            "--execute",
        ]
    if pipeline == "neuroprep-susceptibility":
        command = [
            sys.executable,
            "-m",
            "neuroprep.cli",
            "preprocess-susceptibility",
            "--bids-root",
            str(dataset.bids_root),
            "--output-root",
            str(dataset.derivatives_root / pipeline),
            "--work-root",
            str(dataset.work_root / pipeline),
            "--subject",
            subject_token,
            "--reference-root",
            str(_anatomical_source_root(dataset, anatomical_pipeline or "smriprep", subject_token)),
            "--execute",
        ]
        template = dataset.metadata.get("standard_template") or dataset.metadata.get("mni_template")
        if template:
            command.extend(["--template", str(template)])
        return _containerize_native_upstream(command, dataset)
    return []


def _containerize_native_upstream(command: list[str], dataset: DatasetConfig) -> list[str]:
    """Run ANTs/SynthStrip executors in a pinned image when configured."""

    images = dataset.metadata.get("pipeline_images", {})
    image = images.get("neuroprep-native") if isinstance(images, dict) else None
    if not image:
        return command
    source_root = Path(__file__).resolve().parents[1]
    python_command = list(command)
    python_command[0] = "python"
    binds = [
        (dataset.bids_root, True),
        (dataset.derivatives_root, False),
        (dataset.work_root, False),
        (source_root, True),
    ]
    for key in ("standard_template", "standard_template_mask", "mni_template"):
        value = dataset.metadata.get(key)
        if value:
            candidate = Path(str(value)).expanduser()
            if all(candidate != path for path, _ in binds):
                binds.append((candidate, True))
    container_command = ["singularity", "exec", "--cleanenv", "--containall"]
    for path, readonly in binds:
        bind = f"{path}:{path}" + (":ro" if readonly else "")
        container_command.extend(["--bind", bind])
    qsmxt_value = dataset.metadata.get("qsmxt_executable")
    if qsmxt_value:
        qsmxt = Path(str(qsmxt_value)).expanduser()
        container_command.extend(
            [
                "--bind",
                f"{qsmxt}:{qsmxt}:ro",
                "--env",
                f"NEUROPREP_QSMXT={qsmxt}",
            ]
        )
    container_command.extend(["--env", f"PYTHONPATH={source_root}", str(image), *python_command])
    return container_command


def _pipeline_derivative_root(dataset: DatasetConfig, pipeline: str) -> Path:
    root = dataset.derivatives_root / pipeline
    return root


def _anatomical_source_root(
    dataset: DatasetConfig, pipeline: str, subject: str | None = None
) -> Path:
    root = _pipeline_derivative_root(dataset, pipeline)
    if pipeline != "deepprep":
        return root
    # DeepPrep is isolated under sub-*/visits/* in the high-throughput
    # profile.  A common pipeline root lets finalizers and native modality
    # executors discover the matching session without collapsing visits.
    return root


def _preprocess_subject_output(root: Path, pipeline: str, subject: str) -> Path:
    if pipeline == "deepprep":
        return root / f"sub-{subject}"
    return root / f"sub-{subject}"


def _remove_cli_option(command: list[str], option: str, values: int = 0) -> list[str]:
    configured = list(command)
    while option in configured:
        index = configured.index(option)
        del configured[index : min(len(configured), index + values + 1)]
    return configured


def _set_cli_option(command: list[str], option: str, value: str) -> list[str]:
    configured = _remove_cli_option(command, option, values=1)
    configured.extend([option, value])
    return configured


def _pipeline_resource_override(dataset: DatasetConfig, pipeline: str) -> dict[str, Any]:
    raw = dataset.metadata.get("pipeline_resources", {})
    if not isinstance(raw, dict) or not isinstance(raw.get(pipeline), dict):
        return {}
    return dict(raw[pipeline])


def _configure_deepprep(
    command: list[str],
    dataset: DatasetConfig,
    include_standard_space: bool,
    has_fmri: bool,
    task_labels: list[str],
) -> list[str]:
    if not command:
        return command
    configured = list(command)
    if not has_fmri and "--anat_only" not in configured:
        configured.append("--anat_only")
    if has_fmri and task_labels:
        configured = _set_cli_option(configured, "--bold_task_type", " ".join(task_labels))
    target_space = (
        str(dataset.metadata.get("model_ready_space", Space.STANDARD.value))
        if include_standard_space
        else "None"
    )
    configured = _set_cli_option(configured, "--bold_volume_space", target_space)
    configured = _set_cli_option(
        configured,
        "--bold_volume_res",
        f"{int(dataset.metadata.get('model_ready_resolution_mm', 2)):02d}",
    )
    configured = _set_cli_option(configured, "--bold_surface_spaces", "None")
    override = _pipeline_resource_override(dataset, "deepprep")
    if "cpus" in override:
        configured = _set_cli_option(configured, "--cpus", str(int(override["cpus"])))
    if "memory_gb" in override:
        configured = _set_cli_option(configured, "--memory", str(int(override["memory_gb"])))
    if dataset.metadata.get("freesurfer_license") and "--fs_license_file" not in configured:
        configured.extend(["--fs_license_file", "/fs_license.txt"])
    return configured


def _configure_deepprep_postprocess(
    command: list[str],
    dataset: DatasetConfig,
    task_labels: list[str],
    include_standard_space: bool,
) -> list[str]:
    if not command:
        return command
    configured = list(command)
    if task_labels:
        configured = _set_cli_option(configured, "--task_id", " ".join(task_labels))
    space = (
        str(dataset.metadata.get("model_ready_space", Space.STANDARD.value))
        if include_standard_space
        else "T1w"
    )
    configured = _set_cli_option(configured, "--space", space)
    configured = _set_cli_option(
        configured,
        "--skip_frame",
        str(int(dataset.metadata.get("fmri_skip_frames", 2))),
    )
    configured = _set_cli_option(
        configured,
        "--bandpass",
        str(dataset.metadata.get("fmri_bandpass_hz", "0.01-0.08")),
    )
    override = _pipeline_resource_override(dataset, "deepprep-postprocess")
    if "cpus" in override:
        configured = _set_cli_option(configured, "--cpus", str(int(override["cpus"])))
    if "memory_gb" in override:
        configured = _set_cli_option(configured, "--memory", str(int(override["memory_gb"])))
    if dataset.metadata.get("freesurfer_license") and "--fs_license_file" not in configured:
        configured.extend(["--fs_license_file", "/fs_license.txt"])
    return configured


def _configure_qsiprep_dwi_only(
    command: list[str],
    *,
    separate_all_dwis: bool = False,
) -> list[str]:
    if not command:
        return command
    configured = _remove_cli_option(command, "--derivatives", values=1)
    if "--dwi-only" not in configured:
        configured.append("--dwi-only")
    # QSIPrep otherwise defaults to T1w even in its DWI-only workflow and runs a
    # redundant SynthSeg anatomical pass.  NeuroPrep's diffusion specialist
    # deliberately keeps corrected DWI in gradient-safe ACPC space; structural
    # processing and cross-modal alignment are owned by their own specialists.
    configured = _set_cli_option(configured, "--anat-modality", "none")
    if separate_all_dwis and "--separate-all-dwis" not in configured:
        configured.append("--separate-all-dwis")
    return configured


def _qsiprep_acquisition_policy(images: list[ImageRecord]) -> dict[str, Any]:
    """Infer whether QSIPrep must keep DWI runs separate.

    QSIPrep may safely combine repeated or reverse-phase acquisitions when their
    voxel grids match. Combining runs with different shapes or affines causes a
    late, expensive failure in its pre-HMC merge. The diffusion specialist
    therefore inspects NIfTI headers per visit and asks QSIPrep to preprocess
    every DWI independently whenever a visit contains incompatible grids.

    Invalid/unreadable headers are left to BIDS validation and QSIPrep rather
    than being guessed from filenames.
    """

    by_session: dict[str, list[ImageRecord]] = defaultdict(list)
    for image in images:
        if image.modality is Modality.DWI:
            by_session[image.session or "__single__"].append(image)

    incompatible_sessions: list[str] = []
    unreadable: list[str] = []
    geometry: dict[str, list[dict[str, Any]]] = {}
    for session, session_images in sorted(by_session.items()):
        headers: list[tuple[tuple[int, ...], np.ndarray]] = []
        records: list[dict[str, Any]] = []
        for image in sorted(session_images, key=lambda item: item.path.as_posix()):
            try:
                payload = nib.load(str(image.path))
                shape = tuple(int(value) for value in payload.shape[:3])
                affine = np.asarray(payload.affine, dtype=float)
            except Exception:
                unreadable.append(str(image.path))
                continue
            headers.append((shape, affine))
            records.append(
                {
                    "path": str(image.path),
                    "shape": list(shape),
                    "voxel_sizes": [
                        round(float(value), 6) for value in nib.affines.voxel_sizes(affine)
                    ],
                }
            )
        geometry[session] = records
        if len(headers) < 2:
            continue
        reference_shape, reference_affine = headers[0]
        if any(
            shape != reference_shape
            or not np.allclose(affine, reference_affine, rtol=1e-5, atol=1e-4)
            for shape, affine in headers[1:]
        ):
            incompatible_sessions.append(session)

    return {
        "mode": ("separate-all-dwis" if incompatible_sessions else "combine-compatible-dwis"),
        "separate_all_dwis": bool(incompatible_sessions),
        "reason": (
            "incompatible DWI voxel grids within one or more visits"
            if incompatible_sessions
            else "DWI voxel grids are compatible within each visit"
        ),
        "incompatible_sessions": incompatible_sessions,
        "unreadable_headers": unreadable,
        "geometry": geometry,
    }


def _configure_smriprep_spaces(
    command: list[str],
    dataset: DatasetConfig,
    include_standard_space: bool,
) -> list[str]:
    if not command:
        return command
    configured = list(command)
    if "--output-spaces" in configured:
        index = configured.index("--output-spaces")
        del configured[index]
        while index < len(configured) and not configured[index].startswith("--"):
            del configured[index]
    if include_standard_space:
        raw_spaces = dataset.metadata.get(
            "structural_output_spaces",
            [
                "T1w",
                "MNI152NLin2009cAsym:res-"
                f"{int(dataset.metadata.get('model_ready_resolution_mm', 2))}",
            ],
        )
        spaces = (
            [str(raw_spaces)]
            if isinstance(raw_spaces, str)
            else [str(value) for value in raw_spaces]
        )
    else:
        spaces = ["T1w"]
    configured.extend(["--output-spaces", *spaces])
    return configured


def _add_anatomical_finalization_task(
    dataset: DatasetConfig,
    registry: PipelineRegistry,
    subject: str,
    subject_token: str,
    structural_id: str,
    source_pipeline: str,
    include_standard_space: bool,
    tasks: list[Task],
    source_root_override: Path | None = None,
) -> tuple[str, Path]:
    task_id = f"{dataset.dataset_id}--{subject_token}--finalize-anatomical"
    output_root = dataset.derivatives_root / "neuroprep-model-ready"
    marker = output_root / f"sub-{subject_token}" / "t1w-finalization.json"
    source_root = source_root_override or _anatomical_source_root(
        dataset, source_pipeline, subject_token
    )
    command = [
        sys.executable,
        "-m",
        "neuroprep.cli",
        "finalize-anatomical",
        "--pipeline-root",
        str(source_root),
        "--source-pipeline",
        source_pipeline,
        "--output-root",
        str(output_root),
        "--subject",
        subject,
        "--marker",
        str(marker),
    ]
    required_spaces: list[str] = []
    if include_standard_space:
        required_spaces.append("MNI152NLin2009cAsym")
        command.extend(["--require-space", "MNI152NLin2009cAsym"])
    resources = _resources_for("neuroprep-anat-finalize", dataset, registry)
    registration_image: str | None = None
    standard_template: str | None = None
    standard_mask: str | None = None
    if source_pipeline == "deepprep" and include_standard_space:
        pipeline_images = dataset.metadata.get("pipeline_images", {})
        if isinstance(pipeline_images, dict):
            registration_image = pipeline_images.get("neuroprep-native") or pipeline_images.get(
                "qsiprep"
            )
        standard_template_value = dataset.metadata.get("standard_template")
        standard_mask_value = dataset.metadata.get("standard_template_mask")
        standard_template = str(standard_template_value) if standard_template_value else None
        standard_mask = str(standard_mask_value) if standard_mask_value else None
        if registration_image and standard_template and standard_mask:
            command.extend(
                [
                    "--registration-container",
                    registration_image,
                    "--standard-template",
                    standard_template,
                    "--standard-mask",
                    standard_mask,
                    "--threads",
                    str(int(resources.get("cpus", 8))),
                ]
            )
    tasks.append(
        Task(
            task_id=task_id,
            dataset_id=dataset.dataset_id,
            stage=TaskStage.POSTPROCESS,
            pipeline="neuroprep-anat-finalize",
            subject=subject,
            dependencies=[structural_id],
            inputs=[str(source_root / f"sub-{subject_token}")],
            outputs=[str(marker)],
            params={
                "skull_stripped_required": True,
                "source_pipeline": source_pipeline,
                "bias_field_correction": (
                    "N4 bias-field correction via DeepPrep/FastSurfer"
                    if source_pipeline == "deepprep"
                    else "N4BiasFieldCorrection via sMRIPrep"
                ),
                "denoising": (
                    "executor-defined; no blanket smoothing"
                    if source_pipeline == "deepprep"
                    else "ANTs DenoiseImage via sMRIPrep"
                ),
                "intensity_normalized": True,
                "required_spaces": [Space.SUBJECT.value, *required_spaces],
                "registration_container": registration_image,
                "standard_template": standard_template,
                "standard_template_mask": standard_mask,
                "registration_threads": int(resources.get("cpus", 8)),
                "output_contract_version": OUTPUT_CONTRACT_VERSION,
                "verify_outputs": True,
            },
            resources=resources,
            command=command,
        )
    )
    return task_id, marker


def _add_modality_finalization_task(
    dataset: DatasetConfig,
    registry: PipelineRegistry,
    subject: str,
    subject_token: str,
    modality: Modality,
    pipeline: str,
    preprocessing_id: str,
    include_standard_space: bool,
    tasks: list[Task],
    pipeline_root_override: Path | None = None,
) -> tuple[str, Path]:
    modality_name = modality.value
    task_id = f"{dataset.dataset_id}--{subject_token}--finalize-{modality_name}"
    marker = (
        dataset.derivatives_root
        / "neuroprep-model-ready"
        / f"sub-{subject_token}"
        / f"{modality_name}-finalization.json"
    )
    output_root = dataset.derivatives_root / "neuroprep-model-ready"
    pipeline_root = pipeline_root_override or (
        _anatomical_source_root(dataset, "deepprep", subject_token)
        if pipeline == "deepprep"
        else _pipeline_derivative_root(dataset, pipeline)
    )
    command = [
        sys.executable,
        "-m",
        "neuroprep.cli",
        "finalize-modality",
        "--pipeline-root",
        str(pipeline_root),
        "--source-pipeline",
        pipeline,
        "--output-root",
        str(output_root),
        "--bids-root",
        str(dataset.bids_root),
        "--subject",
        subject,
        "--modality",
        modality_name,
        "--marker",
        str(marker),
    ]
    support_roots: list[Path] = []
    if modality is Modality.FMRI and pipeline == "xcpd":
        support_roots.append(dataset.derivatives_root / "fmriprep")
    elif modality is Modality.FMRI and pipeline in {"deepprep", "deepprep-postprocess"}:
        support_roots.append(
            (
                _pipeline_derivative_root(dataset, "deepprep")
                if pipeline_root_override is not None
                else _anatomical_source_root(dataset, "deepprep", subject_token)
            )
        )
    for root in support_roots:
        command.extend(["--support-root", str(root)])
    required_spaces: list[str] = []
    if modality is Modality.DWI:
        required_spaces.append("ACPC")
    elif include_standard_space and modality in {
        Modality.T2W,
        Modality.FLAIR,
        Modality.PD,
        Modality.FMRI,
        Modality.PET,
        Modality.ASL,
        Modality.MRA,
        Modality.CT,
    }:
        required_spaces.append(Space.STANDARD.value)
    for space in required_spaces:
        command.extend(["--require-space", space])
    tasks.append(
        Task(
            task_id=task_id,
            dataset_id=dataset.dataset_id,
            stage=TaskStage.POSTPROCESS,
            pipeline="neuroprep-modality-finalize",
            subject=subject,
            dependencies=[preprocessing_id],
            inputs=[
                str(pipeline_root / f"sub-{subject_token}"),
                *[str(root / f"sub-{subject_token}") for root in support_roots],
            ],
            outputs=[str(marker)],
            params={
                "modalities": [modality_name],
                "source_pipeline": pipeline,
                "support_roots": [str(root) for root in support_roots],
                "mask_required": modality not in ELECTROPHYS_MODALITIES,
                "required_spaces": required_spaces,
                "output_contract_version": OUTPUT_CONTRACT_VERSION,
                "verify_outputs": True,
            },
            resources=_resources_for("neuroprep-modality-finalize", dataset, registry),
            command=command,
        )
    )
    return task_id, marker


def _resources_for(
    pipeline: str,
    dataset: DatasetConfig,
    registry: PipelineRegistry,
) -> dict[str, Any]:
    resources = registry.resources(pipeline)
    raw_overrides = dataset.metadata.get("pipeline_resources", {})
    if isinstance(raw_overrides, dict) and isinstance(raw_overrides.get(pipeline), dict):
        resources.update(raw_overrides[pipeline])
    return resources


def _add_registration_task(
    dataset: DatasetConfig,
    registry: PipelineRegistry,
    subject: str,
    subject_token: str,
    images: list[ImageRecord],
    modalities: set[Modality],
    preprocessing_ids: list[str],
    target_spaces: list[str],
    include_standard_space: bool,
    tasks: list[Task],
    warnings: list[str],
) -> str | None:
    spatial = modalities & SPATIAL_MODALITIES
    needs_spatial_alignment = len(spatial) > 1
    needs_sensor_alignment = bool(modalities & ELECTROPHYS_MODALITIES) and bool(
        modalities & STRUCTURAL_MODALITIES
    )
    # Sensor recordings are not voxel images. Their digitization/electrode coordinates are
    # validated (and, when configured, transformed) at the common-alignment boundary instead of
    # creating a fictitious image-registration task here.
    if not needs_spatial_alignment:
        return None
    task_id = f"{dataset.dataset_id}--{subject_token}--register"
    reference = select_reference(images)
    if reference is None:
        warnings.append(f"No anatomical reference could be selected for subject {subject}")
    command = registry.command("neuroprep-register", dataset, subject)
    output = (
        dataset.derivatives_root / "neuroprep-registration" / f"sub-{subject_token}_transforms.json"
    )
    search_roots = [
        _preprocess_subject_output(
            _pipeline_derivative_root(dataset, pipeline), pipeline, subject_token
        )
        for pipeline in sorted(
            {
                task.pipeline
                for task in tasks
                if task.subject == subject and task.stage is TaskStage.PREPROCESS
            }
        )
    ]
    if not command:
        command = [
            sys.executable,
            "-m",
            "neuroprep.cli",
            "index-transforms",
            "--subject",
            subject,
            "--output",
            str(output),
            "--search",
            *[str(path) for path in search_roots],
        ]
    tasks.append(
        Task(
            task_id=task_id,
            dataset_id=dataset.dataset_id,
            stage=TaskStage.REGISTER,
            pipeline="neuroprep-register",
            subject=subject,
            dependencies=preprocessing_ids,
            inputs=[str(image.path) for image in images],
            outputs=[str(output)],
            params={
                "reference": str(reference.path) if reference else None,
                "reference_modality": reference.modality.value if reference else None,
                "spaces": target_spaces,
                "standard_space_enabled": include_standard_space,
                "resample_policy": "compose-transforms-and-resample-once",
                "sensor_alignment_stage": "common-alignment-coordinate-validation"
                if needs_sensor_alignment
                else None,
                "sensor_alignment_required": needs_sensor_alignment,
                "output_contract_version": OUTPUT_CONTRACT_VERSION,
                "verify_outputs": True,
            },
            resources=_resources_for("neuroprep-register", dataset, registry),
            command=command,
        )
    )
    return task_id


def _add_qc_task(
    dataset: DatasetConfig,
    registry: PipelineRegistry,
    subject: str,
    subject_token: str,
    images: list[ImageRecord],
    registration_id: str | None,
    dependencies: list[str],
    finalization_markers: list[Path],
    tasks: list[Task],
) -> str:
    electrophysiology = any(image.modality in ELECTROPHYS_MODALITIES for image in images)
    pipeline = "neuroprep-qc-multimodal" if electrophysiology else "neuroprep-qc"
    output = dataset.derivatives_root / "neuroprep-qc" / f"sub-{subject_token}.json"
    command = registry.command(pipeline, dataset, subject)
    if not command and finalization_markers:
        command = [
            sys.executable,
            "-m",
            "neuroprep.cli",
            "production-qc",
            "--finalization-markers",
            *[str(path) for path in finalization_markers],
            "--output",
            str(output),
        ]
        if dataset.metadata.get("qc_thresholds"):
            command.extend(["--thresholds", str(dataset.metadata["qc_thresholds"])])
        command.extend(
            [
                "--review-policy",
                str(dataset.metadata.get("qc_review_policy", "block")),
            ]
        )
    task_id = f"{dataset.dataset_id}--{subject_token}--qc"
    tasks.append(
        Task(
            task_id=task_id,
            dataset_id=dataset.dataset_id,
            stage=TaskStage.QC,
            pipeline=pipeline,
            subject=subject,
            dependencies=list(dict.fromkeys(dependencies)),
            inputs=[str(path) for path in finalization_markers],
            outputs=[str(output)],
            params={
                "require_registration_overlays": registration_id is not None,
                "output_contract_version": OUTPUT_CONTRACT_VERSION,
                "release_gate": True,
                "review_policy": str(dataset.metadata.get("qc_review_policy", "block")),
            },
            resources=_resources_for(pipeline, dataset, registry),
            command=command,
        )
    )
    return task_id


def _add_model_ready_task(
    dataset: DatasetConfig,
    registry: PipelineRegistry,
    subject: str,
    subject_token: str,
    finalization_markers: list[Path],
    qc_id: str,
    tasks: list[Task],
) -> str:
    task_id = f"{dataset.dataset_id}--{subject_token}--model-ready"
    qc_report = dataset.derivatives_root / "neuroprep-qc" / f"sub-{subject_token}.json"
    output = (
        dataset.derivatives_root
        / "neuroprep-model-ready"
        / f"sub-{subject_token}"
        / f"sub-{subject_token}_model-ready.json"
    )
    command = [
        sys.executable,
        "-m",
        "neuroprep.cli",
        "model-ready-subject",
        "--dataset-id",
        dataset.dataset_id,
        "--subject",
        subject,
        "--finalization-markers",
        *[str(path) for path in finalization_markers],
        "--qc-report",
        str(qc_report),
        "--output",
        str(output),
    ]
    if not finalization_markers:
        command = []
    tasks.append(
        Task(
            task_id=task_id,
            dataset_id=dataset.dataset_id,
            stage=TaskStage.RELEASE,
            pipeline="neuroprep-model-ready",
            subject=subject,
            dependencies=[qc_id],
            inputs=[str(path) for path in finalization_markers] + [str(qc_report)],
            outputs=[str(output)],
            params={
                "output_kind": "model-ready-subject-manifest",
                "output_contract_version": OUTPUT_CONTRACT_VERSION,
                "verify_outputs": True,
                "fail_closed": True,
            },
            resources=_resources_for("neuroprep-model-ready", dataset, registry),
            command=command,
        )
    )
    return task_id


def _add_dataset_model_ready_index_task(
    dataset: DatasetConfig,
    registry: PipelineRegistry,
    subject_release_ids: list[str],
    tasks: list[Task],
) -> str:
    task_id = f"{dataset.dataset_id}--model-ready-index"
    output_root = dataset.derivatives_root / "neuroprep-model-ready"
    jsonl = output_root / "index" / "samples.jsonl"
    tsv = output_root / "index" / "samples.tsv"
    tasks.append(
        Task(
            task_id=task_id,
            dataset_id=dataset.dataset_id,
            stage=TaskStage.INDEX,
            pipeline="neuroprep-model-ready-index",
            dependencies=subject_release_ids,
            inputs=[str(output_root)],
            outputs=[str(jsonl), str(tsv)],
            params={
                "output_kind": "model-ready-dataset-index",
                "output_contract_version": OUTPUT_CONTRACT_VERSION,
                "verify_outputs": True,
                "fail_closed": True,
            },
            resources=_resources_for("neuroprep-model-ready-index", dataset, registry),
            command=[
                sys.executable,
                "-m",
                "neuroprep.cli",
                "model-ready-index",
                "--root",
                str(output_root),
                "--output-jsonl",
                str(jsonl),
                "--output-tsv",
                str(tsv),
            ],
        )
    )
    return task_id


def _add_dataset_training_tasks(
    dataset: DatasetConfig,
    registry: PipelineRegistry,
    index_id: str,
    tasks: list[Task],
    warnings: list[str],
    has_spatial_modalities: bool,
) -> None:
    model_root = dataset.derivatives_root / "neuroprep-model-ready"
    sample_index = model_root / "index" / "samples.jsonl"
    training_root = dataset.derivatives_root / "neuroprep-training"
    reference = training_root / "grid" / "common_grid_reference.nii.gz"
    visits_root = training_root / "visits"
    visit_index = visits_root / "index" / "visits.jsonl"
    aligned_root = training_root / "aligned"
    aligned_index = aligned_root / "index" / "aligned-visits.jsonl"
    tensor_root = training_root / "tensors"
    tensor_index = tensor_root / "index" / "tensors.jsonl"

    grid_id: str | None = None
    if has_spatial_modalities:
        grid_id = f"{dataset.dataset_id}--common-grid"
        tasks.append(
            Task(
                task_id=grid_id,
                dataset_id=dataset.dataset_id,
                stage=TaskStage.GRID,
                pipeline="neuroprep-common-grid",
                dependencies=[index_id],
                inputs=[str(sample_index)],
                outputs=[str(reference), str(reference.with_name(reference.name[:-7] + ".json"))],
                params={
                    "space": str(dataset.metadata.get("model_ready_space", Space.STANDARD.value)),
                    "selection": "deterministic passing standard-space anatomical sample",
                    "output_contract_version": OUTPUT_CONTRACT_VERSION,
                    "verify_outputs": True,
                },
                resources=_resources_for("neuroprep-common-grid", dataset, registry),
                command=[
                    sys.executable,
                    "-m",
                    "neuroprep.cli",
                    "grid-reference",
                    "--sample-index",
                    str(sample_index),
                    "--output",
                    str(reference),
                    "--required-space",
                    str(dataset.metadata.get("model_ready_space", Space.STANDARD.value)),
                ],
            )
        )

    assemble_id = f"{dataset.dataset_id}--assemble-visits"
    expected = dataset.metadata.get(
        "expected_modalities",
        [
            modality.value
            for modality in Modality
            if modality not in {Modality.UNKNOWN, Modality.FIELDMAP}
        ],
    )
    expected_modalities = (
        [str(expected)] if isinstance(expected, str) else [str(value) for value in expected]
    )
    tasks.append(
        Task(
            task_id=assemble_id,
            dataset_id=dataset.dataset_id,
            stage=TaskStage.ASSEMBLE,
            pipeline="neuroprep-visit-assembly",
            dependencies=[index_id],
            inputs=[str(sample_index)],
            outputs=[str(visit_index), str(visits_root / "index" / "visits.tsv")],
            params={
                "expected_modalities": expected_modalities,
                "visit_tolerance_days": int(dataset.metadata.get("visit_tolerance_days", 30)),
                "output_contract_version": "3.0",
                "verify_outputs": True,
            },
            resources=_resources_for("neuroprep-visit-assembly", dataset, registry),
            command=[
                sys.executable,
                "-m",
                "neuroprep.cli",
                "assemble-visits",
                "--sample-index",
                str(sample_index),
                "--output-root",
                str(visits_root),
                "--bids-root",
                str(dataset.bids_root),
                "--dataset-id",
                dataset.dataset_id,
                "--expected-modalities",
                *expected_modalities,
                "--tolerance-days",
                str(int(dataset.metadata.get("visit_tolerance_days", 30))),
            ],
        )
    )

    align_id = f"{dataset.dataset_id}--align-visits"
    align_command = [
        sys.executable,
        "-m",
        "neuroprep.cli",
        "align-dataset",
        "--visit-index",
        str(visit_index),
        "--output-root",
        str(aligned_root),
        "--output-index",
        str(aligned_index),
    ]
    if has_spatial_modalities:
        align_command.extend(["--reference", str(reference)])
    sensor_transform_root = dataset.metadata.get("sensor_transform_root")
    if sensor_transform_root:
        align_command.extend(["--sensor-transform-root", str(sensor_transform_root)])
    if bool(dataset.metadata.get("require_standard_sensor_coordinates", False)):
        align_command.append("--require-standard-sensor-coordinates")
    tasks.append(
        Task(
            task_id=align_id,
            dataset_id=dataset.dataset_id,
            stage=TaskStage.ALIGN,
            pipeline="neuroprep-common-grid",
            dependencies=[value for value in (grid_id, assemble_id) if value],
            inputs=[str(visit_index), *([str(reference)] if has_spatial_modalities else [])],
            outputs=[str(aligned_index)],
            params={
                "dwi_policy": "derive shell scalar maps; never resample raw gradients",
                "sensor_policy": "validate or explicitly transform coordinates",
                "voxel_grid_required": has_spatial_modalities,
                "single_resampling": True,
                "output_contract_version": "3.0",
                "verify_outputs": True,
            },
            resources=_resources_for("neuroprep-common-grid", dataset, registry),
            command=align_command,
        )
    )

    tensor_id = f"{dataset.dataset_id}--tensorize-visits"
    target_shape = dataset.metadata.get("tensor_shape", [96, 112, 96])
    dwi_shells = dataset.metadata.get("dwi_shells", [0, 1000, 2000, 3000])
    tensor_command = [
        sys.executable,
        "-m",
        "neuroprep.cli",
        "tensorize-dataset",
        "--aligned-index",
        str(aligned_index),
        "--output-root",
        str(tensor_root),
        "--output-index",
        str(tensor_index),
        "--target-shape",
        *[str(value) for value in target_shape],
        "--dwi-shells",
        *[str(value) for value in dwi_shells],
        "--fmri-window",
        str(int(dataset.metadata.get("fmri_window", 100))),
        "--fmri-stride",
        str(int(dataset.metadata.get("fmri_stride", 50))),
    ]
    tasks.append(
        Task(
            task_id=tensor_id,
            dataset_id=dataset.dataset_id,
            stage=TaskStage.TENSOR,
            pipeline="neuroprep-tensors",
            dependencies=[align_id],
            inputs=[str(aligned_index)],
            outputs=[str(tensor_index)],
            params={
                "target_shape": list(target_shape),
                "dwi_shells": list(dwi_shells),
                "missing_modality_mask": True,
                "frameworks": ["numpy", "pytorch", "monai-compatible"],
                "output_contract_version": "3.0",
                "verify_outputs": True,
            },
            resources=_resources_for("neuroprep-tensors", dataset, registry),
            command=tensor_command,
        )
    )
    _add_optional_phenotype_tasks(
        dataset, registry, tensor_id, tensor_index, training_root, tasks, warnings
    )


def _add_optional_phenotype_tasks(
    dataset: DatasetConfig,
    registry: PipelineRegistry,
    tensor_id: str,
    tensor_index: Path,
    training_root: Path,
    tasks: list[Task],
    warnings: list[str],
) -> None:
    phenotype_table = dataset.metadata.get("phenotype_table")
    if not phenotype_table:
        warnings.append(
            "Phenotype linkage is available but not scheduled because metadata.phenotype_table "
            "is not configured"
        )
        return
    phenotype_root = training_root / "phenotypes"
    joined = phenotype_root / "joined-visits.jsonl"
    join_report = phenotype_root / "join-report.json"
    labels = [str(value) for value in dataset.metadata.get("label_columns", [])]
    covariates = [str(value) for value in dataset.metadata.get("covariate_columns", [])]
    command = [
        sys.executable,
        "-m",
        "neuroprep.cli",
        "phenotype-join",
        "--visit-index",
        str(tensor_index),
        "--phenotype-table",
        str(phenotype_table),
        "--output",
        str(joined),
        "--report",
        str(join_report),
        "--participant-column",
        str(dataset.metadata.get("phenotype_participant_column", "participant_id")),
    ]
    if labels:
        command.extend(["--label-columns", *labels])
    if covariates:
        command.extend(["--covariate-columns", *covariates])
    session_column = dataset.metadata.get("phenotype_session_column")
    date_column = dataset.metadata.get("phenotype_date_column")
    if session_column:
        command.extend(["--session-column", str(session_column)])
    if date_column:
        command.extend(["--date-column", str(date_column)])
    join_id = f"{dataset.dataset_id}--link-phenotypes"
    tasks.append(
        Task(
            task_id=join_id,
            dataset_id=dataset.dataset_id,
            stage=TaskStage.LINK,
            pipeline="neuroprep-phenotype-link",
            dependencies=[tensor_id],
            inputs=[str(tensor_index), str(phenotype_table)],
            outputs=[str(joined), str(join_report)],
            params={
                "label_columns": labels,
                "covariate_columns": covariates,
                "privacy": "allow-list-only",
                "output_contract_version": "3.0",
                "verify_outputs": True,
            },
            resources={"cpus": 1, "memory_gb": 8, "time": "02:00:00"},
            command=command,
        )
    )
    split_output = phenotype_root / "folds.jsonl"
    split_report = phenotype_root / "split-report.json"
    split_id = f"{dataset.dataset_id}--make-splits"
    split_command = [
        sys.executable,
        "-m",
        "neuroprep.cli",
        "make-splits",
        "--joined-index",
        str(joined),
        "--output",
        str(split_output),
        "--report",
        str(split_report),
        "--folds",
        str(int(dataset.metadata.get("split_folds", 5))),
        "--site-column",
        str(dataset.metadata.get("site_column", "site")),
        "--strategy",
        str(dataset.metadata.get("split_strategy", "stratified-group")),
    ]
    if labels:
        split_command.extend(["--label-column", labels[0]])
    tasks.append(
        Task(
            task_id=split_id,
            dataset_id=dataset.dataset_id,
            stage=TaskStage.SPLIT,
            pipeline="neuroprep-grouped-splits",
            dependencies=[join_id],
            inputs=[str(joined)],
            outputs=[str(split_output), str(split_report)],
            params={
                "participant_grouped": True,
                "site_stratified": True,
                "output_contract_version": "3.0",
                "verify_outputs": True,
            },
            resources={"cpus": 1, "memory_gb": 8, "time": "02:00:00"},
            command=split_command,
        )
    )
    harmonization_input = dataset.metadata.get("harmonization_input")
    feature_columns = [
        str(value) for value in dataset.metadata.get("harmonization_feature_columns", [])
    ]
    if harmonization_input and feature_columns:
        harmonization_root = training_root / "harmonization"
        harmonized_output = harmonization_root / "harmonized-folds.tsv"
        harmonization_report = harmonization_root / "harmonization-report.json"
        harmonize_id = f"{dataset.dataset_id}--harmonize-folds"
        harmonize_command = [
            sys.executable,
            "-m",
            "neuroprep.cli",
            "harmonize-folds",
            "--input",
            str(harmonization_input),
            "--output",
            str(harmonized_output),
            "--output-root",
            str(harmonization_root / "models"),
            "--report",
            str(harmonization_report),
            "--participant-column",
            str(dataset.metadata.get("harmonization_participant_column", "participant_id")),
            "--fold-column",
            str(dataset.metadata.get("harmonization_fold_column", "fold")),
            "--split-column",
            str(dataset.metadata.get("harmonization_split_column", "split")),
            "--batch-column",
            str(dataset.metadata.get("harmonization_batch_column", "site")),
            "--feature-columns",
            *feature_columns,
        ]
        protected = [
            str(value) for value in dataset.metadata.get("harmonization_protected_columns", [])
        ]
        if protected:
            harmonize_command.extend(["--protected-columns", *protected])
        tasks.append(
            Task(
                task_id=harmonize_id,
                dataset_id=dataset.dataset_id,
                stage=TaskStage.HARMONIZE,
                pipeline="neuroprep-fold-harmonization",
                dependencies=[split_id],
                inputs=[str(harmonization_input), str(split_output)],
                outputs=[str(harmonized_output), str(harmonization_report)],
                params={
                    "fit_scope": "training-only-per-fold",
                    "frozen_apply": ["validation", "test"],
                    "participant_grouping_required": True,
                    "feature_columns": feature_columns,
                    "output_contract_version": "3.0",
                    "verify_outputs": True,
                },
                resources={"cpus": 4, "memory_gb": 16, "time": "04:00:00"},
                command=harmonize_command,
            )
        )
    elif harmonization_input or feature_columns:
        warnings.append(
            "Fold harmonization was not scheduled because both metadata.harmonization_input "
            "and metadata.harmonization_feature_columns are required"
        )


def _finalize_tasks(tasks: list[Task]) -> None:
    _validate_dependencies(tasks)
    for task in tasks:
        payload = task.to_dict()
        payload["fingerprint"] = ""
        task.fingerprint = task_fingerprint(payload)


def _validate_dependencies(tasks: list[Task]) -> None:
    task_ids = {task.task_id for task in tasks}
    if len(task_ids) != len(tasks):
        raise ValueError("Plan contains duplicate task identifiers")
    for task in tasks:
        missing = set(task.dependencies) - task_ids
        if missing:
            raise ValueError(f"Task {task.task_id} has missing dependencies: {sorted(missing)}")
