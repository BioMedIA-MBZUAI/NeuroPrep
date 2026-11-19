from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .models import DatasetConfig, Modality


DEFAULT_EXECUTION_PROFILE = "high-throughput"


@dataclass(frozen=True)
class ExecutionProfile:
    """Versioned routing policy for scientific executors.

    Profiles change which maintained upstream application performs a scientific
    step. They never bypass BIDS conversion, BIDS validation, acquisition
    auditing, QC, or the common NeuroPrep release boundary.
    """

    name: str
    version: str
    t1w_pipeline: str
    fmri_pipeline: str
    fmri_rest_postprocess: str | None
    dwi_pipeline: str
    reuse_external_anatomy: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROFILES: dict[str, ExecutionProfile] = {
    "high-throughput": ExecutionProfile(
        name="high-throughput",
        version="1.0",
        t1w_pipeline="deepprep",
        fmri_pipeline="deepprep",
        fmri_rest_postprocess="deepprep-postprocess",
        dwi_pipeline="qsiprep",
        reuse_external_anatomy=False,
        description=(
            "GPU-accelerated T1w/fMRI processing with DeepPrep, CUDA QSIPrep for diffusion, "
            "and the standard NeuroPrep QC/alignment/tensor boundary."
        ),
    ),
    "reference": ExecutionProfile(
        name="reference",
        version="1.0",
        t1w_pipeline="smriprep",
        fmri_pipeline="fmriprep",
        fmri_rest_postprocess="xcpd",
        dwi_pipeline="qsiprep",
        reuse_external_anatomy=True,
        description=(
            "Compatibility profile using sMRIPrep, fMRIPrep/XCP-D, and QSIPrep. "
            "This is slower and is retained for reference validation and acquisitions "
            "not yet accepted by the high-throughput profile."
        ),
    ),
}


def execution_profile(dataset: DatasetConfig) -> ExecutionProfile:
    raw = str(dataset.metadata.get("execution_profile", DEFAULT_EXECUTION_PROFILE))
    name = raw.strip().lower().replace("_", "-")
    aliases = {
        "fast": "high-throughput",
        "throughput": "high-throughput",
        "highthroughput": "high-throughput",
        "nipreps": "reference",
        "validated-reference": "reference",
    }
    name = aliases.get(name, name)
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown execution profile {raw!r}; choose one of: {choices}") from exc


def is_resting_fmri(images: list[Any], dataset: DatasetConfig) -> bool:
    """Return whether DeepPrep's resting-state postprocessor is appropriate.

    The policy is intentionally conservative: task fMRI is preprocessed but is
    not band-pass filtered or nuisance-regressed as if it were resting-state
    data. A dataset may explicitly declare ``fmri_analysis: resting-state`` when
    task entities are absent.
    """

    declared = str(dataset.metadata.get("fmri_analysis", "auto")).strip().lower()
    if declared in {"rest", "resting", "resting-state", "rsfmri"}:
        return True
    if declared in {"task", "task-fmri", "task-based"}:
        return False
    fmri = [image for image in images if image.modality is Modality.FMRI]
    if not fmri:
        return False
    task_labels = {
        str(image.entities.get("task", "")).strip().lower()
        for image in fmri
        if image.entities.get("task")
    }
    return bool(task_labels) and task_labels <= {"rest", "resting", "restingstate", "rsfmri"}


def fmri_task_labels(images: list[Any]) -> list[str]:
    return sorted(
        {
            str(image.entities["task"])
            for image in images
            if image.modality is Modality.FMRI and image.entities.get("task")
        }
    )
