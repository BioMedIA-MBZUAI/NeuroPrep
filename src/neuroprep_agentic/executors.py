from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Iterable

from . import __version__
from .execution_models import (
    ArtifactRequirement,
    ExecutionScope,
    ExecutorBackend,
    ExecutorDefinition,
)
from .registry import PluginRegistry


def artifact(kind: str, *patterns: str, required: bool = True) -> ArtifactRequirement:
    return ArtifactRequirement(kind=kind, patterns=patterns, required=required)


def native(
    executor_id: str,
    version: str,
    scope: ExecutionScope,
    plugin_ids: tuple[str, ...],
    stages: tuple[str, ...],
    kind: str,
    artifacts: tuple[ArtifactRequirement, ...],
    *,
    required_metadata: tuple[str, ...] = (),
    notes: str = "",
) -> ExecutorDefinition:
    return ExecutorDefinition(
        executor_id=executor_id,
        version=version,
        backend=ExecutorBackend.NATIVE,
        scope=scope,
        plugin_ids=plugin_ids,
        stages=stages,
        command=(
            "{agentic_python}",
            "-m",
            "neuroprep_agentic.native",
            "run",
            "--kind",
            kind,
            "--context",
            "{context_path}",
        ),
        artifacts=artifacts,
        required_metadata=required_metadata,
        source="NeuroPrep-agentic deterministic native executor",
        notes=notes,
    )


def builtin_executor_definitions() -> tuple[ExecutorDefinition, ...]:
    v4 = "{neuroprep_project}/.venv/bin/neuroprep"
    definitions: list[ExecutorDefinition] = [
        native(
            "neuroprep-bids-stage",
            __version__,
            ExecutionScope.ACQUISITION,
            ("core.bids",),
            ("bids_stage",),
            "bids-stage",
            (artifact("bids_source", "{bids_path}"),),
        ),
        native(
            "neuroprep-bids-init",
            __version__,
            ExecutionScope.DATASET,
            ("core.bids",),
            ("bids_init",),
            "bids-init",
            (artifact("bids_dataset", "{bids_root}/dataset_description.json"),),
        ),
        ExecutorDefinition(
            executor_id="bids-validator",
            version="3.0.1",
            backend=ExecutorBackend.NATIVE,
            scope=ExecutionScope.DATASET,
            plugin_ids=("core.bids",),
            stages=("bids_validate",),
            command=(
                "{agentic_python}",
                "-m",
                "neuroprep_agentic.native",
                "run",
                "--kind",
                "bids-validate",
                "--context",
                "{context_path}",
            ),
            image="{container_root}/bids-validator-3.0.1.sif",
            image_source="docker://bids/validator:3.0.1",
            artifacts=(
                artifact(
                    "bids_validation",
                    "{derivatives_root}/bids-validation/bids-validator.json",
                ),
            ),
            source="https://github.com/bids-standard/bids-validator",
            notes=(
                "Validates BIDS-core content after staging; provisional extension "
                "paths are isolated with .bidsignore."
            ),
        ),
        ExecutorDefinition(
            executor_id="smriprep",
            version="0.19.1",
            backend=ExecutorBackend.BIDS_APP,
            scope=ExecutionScope.SUBJECT,
            plugin_ids=("mri.t1w",),
            stages=("anat_preprocess",),
            image="{container_root}/smriprep-0.19.1.sif",
            image_source="docker://nipreps/smriprep:0.19.1",
            command=(
                "{container_runtime}",
                "run",
                "--cleanenv",
                "--containall",
                "--bind",
                "{bids_root}:/source-data:ro",
                "--bind",
                "{task_work_dir}/smriprep-bids:/data:ro",
                "--bind",
                "{derivatives_root}:/out",
                "--bind",
                "{work_root}:/work",
                "--bind",
                "{freesurfer_license}:/fs_license.txt:ro",
                "{image}",
                "/data",
                "/out",
                "participant",
                "--participant-label",
                "{subject}",
                "--work-dir",
                "/work/smriprep/{subject}",
                "--fs-license-file",
                "/fs_license.txt",
                "--fs-no-reconall",
                "--output-spaces",
                "T1w",
                "MNI152NLin2009cAsym:res-1",
            ),
            environment={
                "TEMPLATEFLOW_HOME": "{work_root}/cache/templateflow",
                "XDG_CACHE_HOME": "{work_root}/cache/xdg",
            },
            artifacts=(
                artifact("preprocessed_t1w", "smriprep/sub-{subject}/**/*desc-preproc_T1w.nii.gz"),
                artifact("brain_mask", "smriprep/sub-{subject}/**/*desc-brain_mask.nii.gz"),
            ),
            source="https://github.com/nipreps/smriprep",
            notes="Canonical structural derivative consumed by downstream BIDS Apps.",
        ),
        ExecutorDefinition(
            executor_id="deepprep",
            version="25.1.0",
            backend=ExecutorBackend.BIDS_APP,
            scope=ExecutionScope.SUBJECT,
            plugin_ids=("mri.t1w",),
            stages=("anat_segment",),
            requires_executor_ids=("smriprep",),
            image="{container_root}/deepprep-25.1.0.sif",
            image_source="docker://pbfslab/deepprep:25.1.0",
            command=(
                "{container_runtime}",
                "run",
                "--nv",
                "--network",
                "none",
                "--cleanenv",
                "--containall",
                "--bind",
                "{bids_root}:/source-data:ro",
                "--bind",
                "{task_work_dir}/deepprep-bids:/data:ro",
                "--bind",
                "{derivatives_root}:/out",
                "--bind",
                "{work_root}:/work",
                "--bind",
                "{freesurfer_license}:/fs_license.txt:ro",
                "{image}",
                "/data",
                "/out/deepprep/sub-{subject}",
                "participant",
                "--participant_label",
                "{subject}",
                "--anat_only",
                "--device",
                "auto",
                "--cpus",
                "{cpus}",
                "--memory",
                "{memory_gb}",
                "--skip_bids_validation",
                "--resume",
            ),
            environment={
                "NXF_HOME": "{work_root}/cache/nextflow",
                "TEMPLATEFLOW_HOME": "{work_root}/cache/templateflow",
                "XDG_CACHE_HOME": "{work_root}/cache/xdg",
                # Singularity --nv injects host GL dispatch libraries ahead of
                # the older container ABI on this cluster. Prefer the image's
                # compatible GL libraries while retaining the NVIDIA driver
                # directories as a fallback for CUDA workloads.
                "LD_LIBRARY_PATH": (
                    "/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:"
                    "/opt/conda/envs/deepprep/lib:/usr/local/nvidia/lib:"
                    "/usr/local/nvidia/lib64:/.singularity.d/libs"
                ),
            },
            artifacts=(
                artifact(
                    "preprocessed_t1w",
                    "deepprep/sub-{subject}/Recon/sub-{subject}/mri/T1.mgz",
                ),
                artifact(
                    "brain_mask",
                    "deepprep/sub-{subject}/Recon/sub-{subject}/mri/brainmask.mgz",
                ),
                artifact(
                    "segmentation",
                    "deepprep/sub-{subject}/Recon/sub-{subject}/mri/aparc+aseg.mgz",
                ),
                artifact(
                    "left_white_surface",
                    "deepprep/sub-{subject}/Recon/sub-{subject}/surf/lh.white",
                ),
                artifact(
                    "right_white_surface",
                    "deepprep/sub-{subject}/Recon/sub-{subject}/surf/rh.white",
                ),
                artifact(
                    "qc_report",
                    "deepprep/sub-{subject}/QC/sub-{subject}/sub-{subject}.html",
                ),
            ),
            source="https://github.com/pBFSLab/DeepPrep",
            notes="Offline network isolation is mandatory; FreeSurfer license is required.",
        ),
        ExecutorDefinition(
            executor_id="fastsurfer",
            version="2.6.1",
            backend=ExecutorBackend.BIDS_APP,
            scope=ExecutionScope.SUBJECT,
            plugin_ids=("mri.t1w",),
            stages=("anat_segment",),
            alternative=True,
            image="{container_root}/fastsurfer-2.6.1.sif",
            image_source="docker://deepmi/fastsurfer:cu126-v2.6.1",
            command=(
                "{container_runtime}",
                "exec",
                "--nv",
                "--cleanenv",
                "--containall",
                "--bind",
                "{bids_root}:/source-data:ro",
                "--bind",
                "{derivatives_root}:/out",
                "--bind",
                "{freesurfer_license}:/fs_license.txt:ro",
                "{image}",
                "/fastsurfer/run_fastsurfer.sh",
                "--t1",
                "{primary_t1w}",
                "--sid",
                "sub-{subject}",
                "--sd",
                "/out/fastsurfer",
                "--fs_license",
                "/fs_license.txt",
                "--threads",
                "{cpus}",
                "--device",
                "auto",
            ),
            artifacts=(
                artifact("preprocessed_t1w", "fastsurfer/sub-{subject}/mri/orig.mgz"),
                artifact("brain_mask", "fastsurfer/sub-{subject}/mri/mask.mgz"),
                artifact(
                    "segmentation",
                    "fastsurfer/sub-{subject}/mri/aparc.DKTatlas+aseg.deep.mgz",
                ),
                artifact("left_white_surface", "fastsurfer/sub-{subject}/surf/lh.white"),
                artifact("right_white_surface", "fastsurfer/sub-{subject}/surf/rh.white"),
            ),
            source="https://github.com/Deep-MI/FastSurfer",
            notes=(
                "Alternative to DeepPrep for whole-brain segmentation and surfaces; "
                "segmentation uses the DKT atlas (DeepPrep: Desikan-Killiany)."
            ),
        ),
        ExecutorDefinition(
            executor_id="synthseg",
            version="2.0-freesurfer-7.4.1",
            backend=ExecutorBackend.BIDS_APP,
            scope=ExecutionScope.SUBJECT,
            plugin_ids=("mri.t1w",),
            stages=("anat_segment",),
            alternative=True,
            image="{container_root}/freesurfer-7.4.1.sif",
            image_source="docker://freesurfer/freesurfer:7.4.1",
            command=(
                "{container_runtime}",
                "exec",
                "--nv",
                "--cleanenv",
                "--containall",
                "--bind",
                "{bids_root}:/source-data:ro",
                "--bind",
                "{derivatives_root}:/out",
                "{image}",
                "/usr/local/freesurfer/bin/mri_synthseg",
                "--i",
                "{primary_t1w}",
                "--o",
                "/out/synthseg/sub-{subject}/sub-{subject}_desc-synthseg_dseg.nii.gz",
                "--parc",
                "--robust",
                "--vol",
                "/out/synthseg/sub-{subject}/sub-{subject}_desc-synthseg_volumes.csv",
                "--qc",
                "/out/synthseg/sub-{subject}/sub-{subject}_desc-synthseg_qc.csv",
                "--threads",
                "{cpus}",
            ),
            environment={"FREESURFER_HOME": "/usr/local/freesurfer"},
            artifacts=(
                artifact(
                    "segmentation", "synthseg/sub-{subject}/sub-{subject}_desc-synthseg_dseg.nii.gz"
                ),
                artifact(
                    "segmentation_volumes",
                    "synthseg/sub-{subject}/sub-{subject}_desc-synthseg_volumes.csv",
                ),
                artifact("qc_scores", "synthseg/sub-{subject}/sub-{subject}_desc-synthseg_qc.csv"),
            ),
            source="https://github.com/BBillot/SynthSeg",
            notes=(
                "Contrast- and resolution-agnostic volumetric segmentation with cortical "
                "parcellation; no surfaces, so it competes only for fast or low-memory requests."
            ),
        ),
        ExecutorDefinition(
            executor_id="basil",
            version="4.0.29-fsl-6.0.7.18",
            backend=ExecutorBackend.NATIVE,
            scope=ExecutionScope.SESSION,
            plugin_ids=("mri.asl",),
            stages=("asl_preprocess", "cbf_quantify"),
            alternative=True,
            image="{container_root}/fsl-6.0.7.18.sif",
            image_source="docker://vnmd/fsl_6.0.7.18:20250928",
            command=(
                "{agentic_python}",
                "-m",
                "neuroprep_agentic.native",
                "run",
                "--kind",
                "basil",
                "--context",
                "{context_path}",
                "--image",
                "{image}",
            ),
            artifacts=(
                artifact(
                    "cbf_map",
                    "native/sub-{subject}/**/basil/basil/native_space/perfusion_calib.nii.gz",
                ),
            ),
            source="https://github.com/physimals/oxford_asl",
            notes="FSL BASIL (oxford_asl) with voxelwise M0 calibration; alternative to ASLPrep.",
        ),
        ExecutorDefinition(
            executor_id="mrtrix3",
            version="3.0.8",
            backend=ExecutorBackend.NATIVE,
            scope=ExecutionScope.SUBJECT,
            plugin_ids=("mri.dwi",),
            stages=("dwi_preprocess", "dti_fit"),
            alternative=True,
            image="{container_root}/mrtrix3-3.0.8.sif",
            image_source="docker://mrtrix3/mrtrix3:3.0.8",
            command=(
                "{agentic_python}",
                "-m",
                "neuroprep_agentic.native",
                "run",
                "--kind",
                "mrtrix3-dwi",
                "--context",
                "{context_path}",
                "--image",
                "{image}",
            ),
            artifacts=(
                artifact(
                    "preprocessed_dwi",
                    "native/sub-{subject}/**/mrtrix3-dwi/*_desc-preproc_dwi.nii.gz",
                ),
                artifact(
                    "dwi_gradients",
                    "native/sub-{subject}/**/mrtrix3-dwi/*_desc-preproc_dwi.bval",
                    "native/sub-{subject}/**/mrtrix3-dwi/*_desc-preproc_dwi.bvec",
                ),
                artifact(
                    "diffusion_model",
                    "native/sub-{subject}/**/mrtrix3-dwi/*_model-tensor_param-fa_dwimap.nii.gz",
                ),
            ),
            source="https://github.com/MRtrix3/mrtrix3",
            notes=(
                "dwidenoise, mrdegibbs, dwifslpreproc (topup with a reverse-PE b0 when present), "
                "dwibiascorrect ants, dwi2mask, tensor fit. Takes over QSIRecon's tensor fit; "
                "QSIRecon's optional models need QSIPrep-layout input and are dropped."
            ),
        ),
        ExecutorDefinition(
            executor_id="totalsegmentator",
            version="2.18.0",
            backend=ExecutorBackend.BIDS_APP,
            scope=ExecutionScope.ACQUISITION,
            plugin_ids=("ct",),
            stages=("ct_segment",),
            alternative=True,
            image="{container_root}/totalsegmentator-2.18.0.sif",
            image_source="docker://wasserth/totalsegmentator:2.18.0",
            command=(
                "{container_runtime}",
                "exec",
                "--nv",
                "--cleanenv",
                "--containall",
                "--bind",
                "{bids_root}:/source-data:ro",
                "--bind",
                "{derivatives_root}:/out",
                "--bind",
                "{work_root}/cache/totalsegmentator:/totalseg-home",
                "{image}",
                "TotalSegmentator",
                "-i",
                "{bids_input}",
                "-o",
                "/out/totalsegmentator/sub-{subject}/{acquisition}_desc-totalseg_dseg.nii.gz",
                "--ml",
                "--task",
                "total",
                "--roi_subset",
                "brain",
                "skull",
                "--device",
                "gpu",
                "--statistics",
            ),
            # Weights are fetched once per run by prepare-assets (runtime_assets.py).
            environment={"TOTALSEG_HOME_DIR": "/totalseg-home"},
            artifacts=(
                artifact(
                    "ct_segmentation",
                    "totalsegmentator/sub-{subject}/*_desc-totalseg_dseg.nii.gz",
                ),
            ),
            source="https://github.com/wasserth/TotalSegmentator",
            notes="Free 'total' task restricted to brain and skull; licensed tasks are not used.",
        ),
        ExecutorDefinition(
            executor_id="fmriprep",
            version="25.2.5",
            backend=ExecutorBackend.BIDS_APP,
            scope=ExecutionScope.SUBJECT,
            plugin_ids=("mri.bold",),
            stages=("bold_preprocess",),
            requires_executor_ids=("smriprep",),
            provides={"fmriprep_bold": "/out/fmriprep"},
            image="{container_root}/fmriprep-25.2.5.sif",
            image_source="docker://nipreps/fmriprep:25.2.5",
            command=(
                "{container_runtime}",
                "run",
                "--cleanenv",
                "--containall",
                "--bind",
                "{bids_root}:/data:ro",
                "--bind",
                "{derivatives_root}:/out",
                "--bind",
                "{work_root}:/work",
                "--bind",
                "{freesurfer_license}:/fs_license.txt:ro",
                "{image}",
                "/data",
                "/out/fmriprep",
                "participant",
                "--participant-label",
                "{subject}",
                "--work-dir",
                "/work/fmriprep/{subject}",
                "--derivatives",
                "smriprep=/out/smriprep",
                "--fs-license-file",
                "/fs_license.txt",
                "--output-spaces",
                "T1w",
                "MNI152NLin2009cAsym:res-2",
                "--skip-bids-validation",
            ),
            environment={
                "TEMPLATEFLOW_HOME": "{work_root}/cache/templateflow",
                "XDG_CACHE_HOME": "{work_root}/cache/xdg",
            },
            artifacts=(
                artifact(
                    "preprocessed_bold", "fmriprep/sub-{subject}/**/*desc-preproc_bold.nii.gz"
                ),
                artifact(
                    "bold_confounds", "fmriprep/sub-{subject}/**/*desc-confounds_timeseries.tsv"
                ),
            ),
            source="https://github.com/nipreps/fmriprep",
        ),
        ExecutorDefinition(
            executor_id="xcp-d",
            version="26.1.1",
            backend=ExecutorBackend.BIDS_APP,
            scope=ExecutionScope.SUBJECT,
            plugin_ids=("mri.bold",),
            stages=("bold_confounds",),
            consumes=("fmriprep_bold",),
            image="{container_root}/xcpd-26.1.1.sif",
            image_source="docker://pennlinc/xcp_d:26.1.1",
            command=(
                "{container_runtime}",
                "run",
                "--cleanenv",
                "--containall",
                "--bind",
                "{derivatives_root}:/out",
                "--bind",
                "{work_root}:/work",
                "{image}",
                "{fmriprep_bold}",
                "/out/xcpd",
                "participant",
                "--participant-label",
                "{subject}",
                "--mode",
                "linc",
                "--file-format",
                "nifti",
                "--fd-thresh",
                "0.3",
                "--min-time",
                "60",
                "--smoothing",
                "0",
                "--output-type",
                "censored",
                "--skip",
                "parcellation",
                "--work-dir",
                "/work/xcpd/{subject}",
            ),
            artifacts=(
                artifact("denoised_bold", "xcpd/sub-{subject}/**/*denoised_bold.nii.gz"),
            ),
            source="https://github.com/PennLINC/xcp_d",
        ),
        ExecutorDefinition(
            executor_id="neuroprep-spatial",
            version="0.5.0",
            backend=ExecutorBackend.CONTAINER,
            scope=ExecutionScope.SUBJECT,
            plugin_ids=("mri.t2w", "mri.flair", "mri.pdw", "ct"),
            stages=(
                "anat_preprocess",
                "anat_segment",
                "flair_preprocess",
                "lesion_ready",
                "ct_preprocess",
                "ct_segment",
            ),
            image="{container_root}/qsiprep-26.0.0.sif",
            image_source="docker://pennlinc/qsiprep:26.0.0",
            command=(
                "{container_runtime}",
                "exec",
                "--cleanenv",
                "--containall",
                "--bind",
                "{bids_root}:/data:ro",
                "--bind",
                "{derivatives_root}:/out",
                "--bind",
                "{work_root}:/work",
                "--bind",
                "{neuroprep_project}/src:/neuroprep-src:ro",
                "--env",
                "PYTHONPATH=/neuroprep-src",
                "--env",
                "TEMPLATEFLOW_HOME=/work/cache/templateflow",
                "--env",
                "XDG_CACHE_HOME=/work/cache/xdg",
                "{image}",
                "python",
                "-m",
                "neuroprep.cli",
                "preprocess-spatial",
                "--bids-root",
                "/data",
                "--output-root",
                "/out/neuroprep-spatial",
                "--work-root",
                "/work/neuroprep-spatial",
                "--subject",
                "{subject}",
                "--modalities",
                "{spatial_modalities}",
                "--execute",
            ),
            artifacts=(
                artifact("preprocessed_spatial", "neuroprep-spatial/sub-{subject}/**/*.nii.gz"),
            ),
            source="NeuroPrep-multimodal-v4 0.5.0",
        ),
        native(
            "neuroprep-quantitative",
            "0.1.0",
            ExecutionScope.ACQUISITION,
            ("mri.qmri",),
            ("qmri_preprocess",),
            "quantitative-imaging",
            (artifact("quantitative_map", "native/sub-{subject}/**/*desc-preproc*.nii.gz"),),
            required_metadata=("Units",),
            notes=(
                "Preserves quantitative units; multi-echo fitting is enabled only "
                "with echo metadata."
            ),
        ),
        native(
            "neuroprep-anatomical-maps",
            "0.1.0",
            ExecutionScope.ACQUISITION,
            ("mri.unit1", "mri.t2star"),
            ("qmri_preprocess", "anat_segment", "t2star_preprocess"),
            "quantitative-imaging",
            (
                artifact(
                    "preprocessed_anatomical_map",
                    "native/sub-{subject}/**/*desc-preproc*.nii.gz",
                ),
            ),
            notes=(
                "Validates conventional native-intensity anatomy without requiring physical units."
            ),
        ),
        ExecutorDefinition(
            executor_id="neuroprep-susceptibility",
            version="0.5.0",
            backend=ExecutorBackend.CONTAINER,
            scope=ExecutionScope.SUBJECT,
            plugin_ids=("mri.swi",),
            stages=("swi_reconstruct", "qsm_estimate"),
            image="{container_root}/qsiprep-26.0.0.sif",
            image_source="docker://pennlinc/qsiprep:26.0.0",
            command=(
                "{container_runtime}",
                "exec",
                "--cleanenv",
                "--containall",
                "--bind",
                "{bids_root}:/data:ro",
                "--bind",
                "{derivatives_root}:/out",
                "--bind",
                "{work_root}:/work",
                "--bind",
                "{neuroprep_project}/src:/neuroprep-src:ro",
                "--env",
                "PYTHONPATH=/neuroprep-src",
                "--env",
                "TEMPLATEFLOW_HOME=/work/cache/templateflow",
                "--env",
                "XDG_CACHE_HOME=/work/cache/xdg",
                "{image}",
                "python",
                "-m",
                "neuroprep.cli",
                "preprocess-susceptibility",
                "--bids-root",
                "/data",
                "--output-root",
                "/out/neuroprep-susceptibility",
                "--work-root",
                "/work/neuroprep-susceptibility",
                "--subject",
                "{subject}",
                "--execute",
            ),
            artifacts=(
                artifact(
                    "susceptibility_derivative",
                    "neuroprep-susceptibility/sub-{subject}/**/*.nii.gz",
                ),
            ),
            required_metadata=("MagneticFieldStrength",),
            source="NeuroPrep-multimodal-v4/QSMxT v9.8.0",
        ),
        ExecutorDefinition(
            executor_id="qsiprep",
            version="26.0.0",
            backend=ExecutorBackend.BIDS_APP,
            scope=ExecutionScope.SUBJECT,
            plugin_ids=("mri.dwi",),
            stages=("dwi_preprocess",),
            requires_executor_ids=("smriprep",),
            provides={"qsiprep_dwi": "/out/qsiprep"},
            image="{container_root}/qsiprep-26.0.0.sif",
            image_source="docker://pennlinc/qsiprep:26.0.0",
            command=(
                "{container_runtime}",
                "run",
                "--nv",
                "--cleanenv",
                "--containall",
                "--bind",
                "{bids_root}:/data:ro",
                "--bind",
                "{derivatives_root}:/out",
                "--bind",
                "{work_root}:/work",
                "--bind",
                "{neuroprep_project}/src/neuroprep/resources:/neuroprep-resources:ro",
                "--bind",
                (
                    "{agentic_resources}/eddy_cuda10.2:"
                    "/app/.pixi/envs/qsiprep/bin/eddy_cuda10.2:ro"
                ),
                "--bind",
                (
                    "{neuroprep_project}/src/neuroprep/resources/qsiprep_gradients_26_0.py:"
                    "/app/.pixi/envs/qsiprep/lib/python3.10/site-packages/qsiprep/"
                    "interfaces/gradients.py:ro"
                ),
                "{image}",
                "/data",
                "/out/qsiprep",
                "participant",
                "--participant-label",
                "{subject}",
                "--work-dir",
                "/work/qsiprep/{subject}",
                "--output-resolution",
                "1.7",
                "--bids-filter-file",
                "/neuroprep-resources/qsiprep_primary_t1.json",
                "--eddy-config",
                "/neuroprep-resources/eddy_cuda.json",
                "--skip-bids-validation",
                "--separate-all-dwis",
            ),
            environment={
                "TEMPLATEFLOW_HOME": "{work_root}/cache/templateflow",
                "LD_LIBRARY_PATH": (
                    "/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:"
                    "/app/.pixi/envs/qsiprep/lib:/usr/local/nvidia/lib:"
                    "/usr/local/nvidia/lib64:/.singularity.d/libs"
                ),
            },
            artifacts=(
                artifact("preprocessed_dwi", "qsiprep/sub-{subject}/**/*desc-preproc_dwi.nii.gz"),
                artifact(
                    "dwi_gradients",
                    "qsiprep/sub-{subject}/**/*desc-preproc_dwi.bval",
                    "qsiprep/sub-{subject}/**/*desc-preproc_dwi.bvec",
                ),
            ),
            required_metadata=("PhaseEncodingDirection",),
            source="https://github.com/PennLINC/qsiprep",
        ),
        ExecutorDefinition(
            executor_id="qsirecon",
            version="1.1.1",
            backend=ExecutorBackend.BIDS_APP,
            scope=ExecutionScope.SUBJECT,
            plugin_ids=("mri.dwi",),
            stages=("dti_fit", "advanced_diffusion_models", "tractography"),
            consumes=("qsiprep_dwi",),
            image="{container_root}/qsirecon-1.1.1.sif",
            image_source="docker://pennlinc/qsirecon:1.1.1",
            command=(
                "{container_runtime}",
                "run",
                "--cleanenv",
                "--containall",
                "--bind",
                "{derivatives_root}:/out",
                "--bind",
                "{work_root}:/work",
                "--bind",
                "{freesurfer_license}:/fs_license.txt:ro",
                "{image}",
                "{qsiprep_dwi}",
                "/out/qsirecon",
                "participant",
                "--participant-label",
                "{subject}",
                "--work-dir",
                "/work/qsirecon/{subject}",
                "--nthreads",
                "{cpus}",
                "--omp-nthreads",
                "4",
                "--fs-license-file",
                "/fs_license.txt",
                "--recon-spec",
                "dsi_studio_gqi",
                "--atlases",
                "AAL116",
            ),
            # Reuse validated templates from the persistent run cache. Preserve the
            # image's own /opt/ants and workbench library paths: overriding
            # LD_LIBRARY_PATH under --cleanenv breaks ANTs in this image.
            environment={
                "TEMPLATEFLOW_HOME": "{work_root}/cache/templateflow",
                "XDG_CACHE_HOME": "{work_root}/cache/xdg",
            },
            artifacts=(artifact("diffusion_model", "qsirecon/**/sub-{subject}/**/*"),),
            source="https://github.com/PennLINC/qsirecon",
            notes=(
                "DSI Studio GQI provides tensor scalars, tractography, and AAL116 "
                "connectivity across diverse schemes."
            ),
        ),
        native(
            "neuroprep-fieldmap-validator",
            "0.1.0",
            ExecutionScope.SESSION,
            ("mri.fmap",),
            ("fieldmap_validate",),
            "fieldmap-validate",
            (artifact("fieldmap_qc", "native/sub-{subject}/**/*fieldmap*_qc.json"),),
            notes="Validates fieldmap inputs; IntendedFor remains optional under BIDS 1.11.",
        ),
        ExecutorDefinition(
            executor_id="aslprep",
            version="26.0.3",
            backend=ExecutorBackend.BIDS_APP,
            scope=ExecutionScope.SUBJECT,
            plugin_ids=("mri.asl",),
            stages=("asl_preprocess", "cbf_quantify"),
            requires_executor_ids=("smriprep",),
            image="{container_root}/aslprep-26.0.3.sif",
            image_source="docker://pennlinc/aslprep:26.0.3",
            command=(
                "{container_runtime}",
                "run",
                "--cleanenv",
                "--containall",
                "--bind",
                "{bids_root}:/data:ro",
                "--bind",
                "{derivatives_root}:/out",
                "--bind",
                "{work_root}:/work",
                "--bind",
                "{freesurfer_license}:/fs_license.txt:ro",
                "--bind",
                (
                    "{work_root}/cache/patches/aslprep-26.0.3/cbf.py:"
                    "/app/.pixi/envs/aslprep/lib/python3.12/site-packages/aslprep/"
                    "workflows/asl/cbf.py:ro"
                ),
                "--bind",
                (
                    "{work_root}/cache/patches/aslprep-26.0.3/base.py:"
                    "/app/.pixi/envs/aslprep/lib/python3.12/site-packages/aslprep/"
                    "workflows/base.py:ro"
                ),
                "{image}",
                "/data",
                "/out/aslprep",
                "participant",
                "--participant-label",
                "{subject}",
                "--work-dir",
                "/work/aslprep/{subject}",
                "--fs-license-file",
                "/fs_license.txt",
                "--fs-no-reconall",
                "--derivatives",
                "smriprep=/out/smriprep",
                "--level",
                "full",
                "--output-spaces",
                "T1w",
                "MNI152NLin2009cAsym:res-2",
                "--skip-bids-validation",
            ),
            environment={
                "TEMPLATEFLOW_HOME": "{work_root}/cache/templateflow",
                "XDG_CACHE_HOME": "{work_root}/cache/xdg",
            },
            artifacts=(artifact("cbf_map", "aslprep/sub-{subject}/**/*cbf.nii.gz"),),
            required_metadata=("ArterialSpinLabelingType",),
            source="https://github.com/PennLINC/aslprep",
        ),
        ExecutorDefinition(
            executor_id="petprep",
            version="0.0.8",
            backend=ExecutorBackend.BIDS_APP,
            scope=ExecutionScope.SUBJECT,
            plugin_ids=("pet",),
            stages=("pet_preprocess", "pet_quantify"),
            requires_executor_ids=("smriprep", "deepprep"),
            image="{container_root}/petprep-0.0.8.sif",
            image_source="docker://nipreps/petprep:0.0.8",
            command=(
                "{container_runtime}",
                "run",
                "--cleanenv",
                "--containall",
                "--bind",
                "{bids_root}:/data:ro",
                "--bind",
                "{derivatives_root}:/out",
                "--bind",
                "{work_root}:/work",
                "--bind",
                "{freesurfer_license}:/fs_license.txt:ro",
                "--bind",
                (
                    "{agentic_resources}/"
                    "petprep_sitecustomize.py:/opt/conda/envs/petprep/lib/python3.12/"
                    "site-packages/sitecustomize.py:ro"
                ),
                "--bind",
                (
                    "{agentic_resources}/"
                    "petprep_nfs_linear.yaml:/petprep-nfs-linear.yaml:ro"
                ),
                "{image}",
                "/data",
                "/out/petprep",
                "participant",
                "--participant-label",
                "{subject}",
                "--work-dir",
                "/work/petprep/nfs-linear-v1/{subject}",
                "--fs-license-file",
                "/fs_license.txt",
                "--fs-subjects-dir",
                "/out/deepprep/sub-{subject}/Recon",
                "--derivatives",
                "smriprep=/out/smriprep",
                "--seg",
                "HOCPA",
                "--use-plugin",
                "/petprep-nfs-linear.yaml",
                "--omp-nthreads",
                "1",
                "--stop-on-first-crash",
                "--notrack",
                "--level",
                "full",
                "--output-spaces",
                "T1w",
                "MNI152NLin2009cAsym:res-2",
            ),
            environment={
                "TEMPLATEFLOW_HOME": "{work_root}/cache/templateflow",
                "XDG_CACHE_HOME": "{work_root}/cache/xdg",
            },
            artifacts=(artifact("preprocessed_pet", "petprep/sub-{subject}/**/*pet.nii.gz"),),
            required_metadata=("TracerName", "Units"),
            source="https://github.com/nipreps/petprep",
        ),
        native(
            executor_id="fsl-mrs",
            version="2.4.17",
            scope=ExecutionScope.SESSION,
            plugin_ids=("mri.mrs",),
            stages=("mrs_preprocess", "metabolite_quantify"),
            kind="mrs",
            artifacts=(artifact("mrs_quantification", "native/sub-{subject}/**/mrs/**/*"),),
            required_metadata=("ResonantNucleus", "SpectrometerFrequency", "BasisPath"),
            notes=(
                "FSL-MRS 2.4.17 is installed from the official FSL conda channel. "
                "BasisPath is mandatory; water-reference and tissue metadata are used when present."
            ),
        ),
    ]

    definitions.extend(_native_and_sensor_definitions(v4))
    return tuple(definitions)


def _native_and_sensor_definitions(v4: str) -> list[ExecutorDefinition]:
    result: list[ExecutorDefinition] = []
    for plugin_id, modality, stage in (
        ("eeg", "eeg", "eeg_preprocess"),
        ("ieeg", "ieeg", "ieeg_preprocess"),
        ("meg", "meg", "meg_preprocess"),
    ):
        result.append(
            ExecutorDefinition(
                executor_id=f"neuroprep-sensor-{modality}",
                version="0.5.0",
                backend=ExecutorBackend.NEUROPREP_V4,
                scope=ExecutionScope.SUBJECT,
                plugin_ids=(plugin_id,),
                stages=(stage,),
                executable=v4,
                command=(
                    v4,
                    "preprocess-sensor",
                    "--bids-root",
                    "{bids_root}",
                    "--output-root",
                    "{derivatives_root}/neuroprep-sensor",
                    "--subject",
                    "{subject}",
                    "--modality",
                    modality,
                    "--execute",
                ),
                artifacts=(
                    artifact(
                        f"preprocessed_{modality}",
                        f"neuroprep-sensor/sub-{{subject}}/**/*{modality}*",
                    ),
                ),
                source="NeuroPrep-multimodal-v4/MNE",
            )
        )

    native_specs = (
        (
            "neuroprep-eeg-features",
            ("eeg", "ieeg", "meg"),
            ("eeg_features", "ieeg_features", "meg_features"),
            "sensor-features",
            "sensor_features",
        ),
        (
            "neuroprep-spect",
            ("spect",),
            ("spect_preprocess",),
            "spect",
            "preprocessed_spect",
        ),
        (
            "neuroprep-angio",
            ("mri.angio",),
            ("angio_preprocess",),
            "angio",
            "preprocessed_angio",
        ),
        (
            "neuroprep-fnirs",
            ("fnirs",),
            ("fnirs_preprocess",),
            "fnirs",
            "preprocessed_fnirs",
        ),
        (
            "neuroprep-microscopy",
            ("microscopy",),
            ("microscopy_preprocess",),
            "microscopy",
            "preprocessed_microscopy",
        ),
        (
            "neuroprep-physiology",
            ("physiology",),
            ("physiology_preprocess",),
            "physiology",
            "preprocessed_physiology",
        ),
        (
            "neuroprep-behavior",
            ("behavior",),
            ("behavior_normalize",),
            "behavior",
            "harmonized_behavior",
        ),
        (
            "neuroprep-genetics",
            ("genetics",),
            ("genetics_validate",),
            "genetics",
            "validated_genetics",
        ),
        (
            "neuroprep-ultrasound",
            ("ultrasound",),
            ("ultrasound_preprocess",),
            "ultrasound",
            "preprocessed_ultrasound",
        ),
        (
            "neuroprep-multimodal-registration",
            ("core.multimodal",),
            ("multimodal_coregistration",),
            "multimodal-registration",
            "multimodal_transforms",
        ),
        (
            "neuroprep-longitudinal-template",
            ("core.longitudinal",),
            ("longitudinal_subject_template",),
            "longitudinal-template",
            "longitudinal_template",
        ),
    )
    for executor_id, plugin_ids, stages, kind, artifact_kind in native_specs:
        if executor_id == "neuroprep-longitudinal-template":
            scope = ExecutionScope.SUBJECT
        elif executor_id == "neuroprep-angio":
            scope = ExecutionScope.ACQUISITION
        else:
            scope = ExecutionScope.SESSION
        result.append(
            native(
                executor_id,
                __version__,
                scope,
                plugin_ids,
                stages,
                kind,
                (artifact(artifact_kind, f"native/sub-{{subject}}/**/{kind}/**/*"),),
            )
        )
    return result


class ExecutorCatalog:
    def __init__(self, definitions: Iterable[ExecutorDefinition]) -> None:
        self._definitions = tuple(definitions)
        self._by_id: dict[str, ExecutorDefinition] = {}
        self._by_route: dict[tuple[str, str], ExecutorDefinition] = {}
        self._alternatives: dict[tuple[str, str], list[ExecutorDefinition]] = {}
        for definition in self._definitions:
            if definition.executor_id in self._by_id:
                raise ValueError(f"duplicate executor id: {definition.executor_id}")
            self._by_id[definition.executor_id] = definition
            for plugin_id in definition.plugin_ids:
                for stage in definition.stages:
                    route = (plugin_id, stage)
                    if definition.alternative:
                        self._alternatives.setdefault(route, []).append(definition)
                        continue
                    if route in self._by_route:
                        previous = self._by_route[route]
                        raise ValueError(
                            f"duplicate executor route {plugin_id}/{stage}: "
                            f"{previous.executor_id}, {definition.executor_id}"
                        )
                    self._by_route[route] = definition
        for route, alternatives in self._alternatives.items():
            if route not in self._by_route:
                names = ", ".join(item.executor_id for item in alternatives)
                raise ValueError(f"alternative executors without a default for {route}: {names}")

    @property
    def definitions(self) -> tuple[ExecutorDefinition, ...]:
        return self._definitions

    def resolve(
        self, plugin_id: str, stage: str, selected: Collection[str] = ()
    ) -> ExecutorDefinition:
        """The alternative a backend decision selected, else the route's default.

        A selected alternative claims every route it serves, so a backend that covers
        several stages (e.g. MRtrix3: preprocessing and tensor fit) runs all of them.
        """
        default, *alternatives = self.alternatives(plugin_id, stage)
        for definition in alternatives:
            if definition.executor_id in selected:
                return definition
        return default

    def alternatives(self, plugin_id: str, stage: str) -> tuple[ExecutorDefinition, ...]:
        """Every executor for a route, default first."""
        try:
            default = self._by_route[(plugin_id, stage)]
        except KeyError as exc:
            raise KeyError(f"no executor for {plugin_id}/{stage}") from exc
        return (default, *self._alternatives.get((plugin_id, stage), ()))

    def by_id(self, executor_id: str) -> ExecutorDefinition:
        try:
            return self._by_id[executor_id]
        except KeyError as exc:
            raise KeyError(f"unknown executor: {executor_id}") from exc

    def audit(self, plugins: PluginRegistry) -> list[str]:
        gaps: list[str] = []
        for plugin in plugins.modalities.values():
            for task in plugin.tasks:
                if (plugin.plugin_id, task.stage) not in self._by_route:
                    gaps.append(f"{plugin.plugin_id}/{task.stage}")
        for plugin_id, stage in (
            ("core.multimodal", "multimodal_coregistration"),
            ("core.longitudinal", "longitudinal_subject_template"),
        ):
            if (plugin_id, stage) not in self._by_route:
                gaps.append(f"{plugin_id}/{stage}")
        return sorted(gaps)

    def digest(self) -> str:
        payload = [item.model_dump(mode="json") for item in self._definitions]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def builtin_executor_catalog() -> ExecutorCatalog:
    return ExecutorCatalog(builtin_executor_definitions())
