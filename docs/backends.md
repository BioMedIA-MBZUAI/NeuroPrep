# Backend comparisons

When a capability has more than one integrated, hardware-compatible backend, the pilot
runs all of them on the same participant and production uses the winner (see
[How it works](how-it-works.md#comparing-alternatives) for the selection rule).

## Validated comparisons (v0.5.0)

These comparisons were validated for v0.5.0 on public OpenNeuro subsets, on Slurm nodes
with one NVIDIA A100-SXM4-40GB GPU, 16 CPU cores and 64 GB of memory, with Qwen3 14B
served locally by Ollama:

| Capability | Backends compared | Pilot result (1 participant) | Selected |
|---|---|---|---|
| Whole-brain segmentation (T1w) | DeepPrep 25.1.0, FastSurfer 2.6.1 | 25.1 vs 40.2 min, both passed | DeepPrep (fastest) |
| ... when speed or low memory is requested | + SynthSeg (FreeSurfer 7.4.1) | SynthSeg 1.7 min | SynthSeg (request) |
| ASL perfusion | ASLPrep 26.0.3, FSL BASIL | 16.8 vs 0.8 min, both passed | ASLPrep (request) |
| CT segmentation | ANTs (NeuroPrep spatial), TotalSegmentator 2.18.0 | 2.3 vs 1.1 min, both passed | TotalSegmentator (request) |
| Diffusion preprocessing | MRtrix3 3.0.8, QSIPrep 26.0.0 (+ QSIRecon) | 26.7 vs 27.2 min, both passed | MRtrix3 (request) |

When a backend fails its pilot it loses instead of failing the run: in an early ASL
validation, ASLPrep failed and BASIL was selected automatically.

!!! note "What a pilot measures"
    Pilots compare pass or fail and runtime only; they do not rank segmentation or
    perfusion accuracy.

## What each backend must produce

A backend only competes when it produces everything its capability promises
(`CAPABILITY_CONTRACTS` in `policy.py`), so a comparison never trades outputs for speed.

| Capability | Compared backends | Note |
|---|---|---|
| Whole-brain segmentation (T1w) | DeepPrep 25.1.0, FastSurfer 2.6.1 | segmentation + white surfaces |
| ... when the request asks for speed or low memory | + SynthSeg (FreeSurfer 7.4.1) | volumetric only, no surfaces |
| ASL perfusion | ASLPrep 26.0.3, FSL BASIL (oxford_asl 4.0.29) | BASIL needs an M0 |
| CT segmentation | NeuroPrep spatial (ANTs), TotalSegmentator 2.18.0 | free `total` task, brain and skull |
| Diffusion preprocessing | QSIPrep 26.0.0, MRtrix3 3.0.8 | MRtrix3 also fits the tensor itself |

## Candidates that are not integrated yet

The backend portfolio records multiple packages per capability. Only pinned, integrated
executors are eligible for selection. Other packages remain explicit qualification
candidates until their adapters and representative smoke tests pass; the agent may plan
those tests but cannot silently promote a candidate.

Diffusion planning also inspects measured q-space sampling. Shelled data uses GPU Eddy;
DSI-like data with more than six nominal nonzero b-value levels uses QSIPrep SHORELine
(`--hmc-model 3dSHORE`) because FSL Eddy does not support DSI. The selected model and
b-value evidence are stored in the task metadata.

How pilot results are signed and checked is described in
[Signed backend evidence](backend-evidence.md).
