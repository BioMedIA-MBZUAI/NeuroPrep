# Supported modalities

| Family | Inputs | Executor |
|---|---|---|
| Structural MRI | T1w, T2w, FLAIR, PDw | sMRIPrep, DeepPrep / FastSurfer / SynthSeg, NeuroPrep spatial |
| Functional MRI | BOLD task and rest | fMRIPrep + XCP-D |
| Diffusion MRI | DWI, DTI, DKI, HARDI, DSI | QSIPrep + QSIRecon, or MRtrix3 |
| Perfusion MRI | ASL | ASLPrep or FSL BASIL |
| Susceptibility and quantitative MRI | SWI, QSM, UNIT1, T2*, qMRI maps | NeuroPrep v4, QSMxT routing |
| Spectroscopy | SVS, MRSI, vendor raw | spec2nii + FSL-MRS |
| PET, SPECT | static and dynamic PET, SPECT | PETPrep; quantitative-preserving validation |
| CT | non-contrast, CTA, perfusion CT | NeuroPrep spatial (ANTs) or TotalSegmentator |
| Electrophysiology | EEG, iEEG, MEG | NeuroPrep v4 MNE pipelines |
| Optical and other | fNIRS, microscopy, ultrasound, physiology, behaviour, genetics | typed native normalization and QC |

The full routing table, with versions and BIDS targets, is in the
[Modality matrix](modality-matrix.md). New modalities are plugins, not changes to core
ingestion.

## Coverage details

The registry covers structural MRI (T1w, T2w, FLAIR, PDw, UNIT1, T2-star, SWI,
quantitative MRI), BOLD fMRI, ASL, DWI/DTI, field maps, MRS, PET, CT, SPECT, EEG, iEEG,
MEG, fNIRS, microscopy, physiology, behavior, genetics and neurosonography.

- **BIDS.** The runtime creates `dataset_description.json`, isolates provisional
  extensions with `.bidsignore`, and runs BIDS Validator 3.0.1 before scientific
  processing.
- **Spectroscopy.** Vendor inputs are converted to anonymized NIfTI-MRS with spec2nii
  before FSL-MRS processing.
- **SPECT.** NIfTI is validated and canonicalized without changing quantitative
  intensities.
- **Ultrasound.** Video is normalized losslessly to FFV1 with explicit stream QC.
- **Behaviour and physiology.** Behaviour JSON/CSV is normalized to TSV, EDF physiology
  is converted to FIF with recording QC, and optional EEG features contain computed
  spectral band powers.

Every registered task route has a concrete deterministic executor definition.
Scientific qualification remains evidence-based: a route is released for a dataset only
after its metadata, container/tool, artifacts and QC contracts pass on that dataset.
Provisional modalities fail closed instead of being presented as validated merely
because they were detected.

## Datasets

Dataset adapters normalize source metadata; modality plugins detect and plan
acquisitions. The same engine therefore handles OASIS4, OASIS3, ADNI, ANMERGE, A4,
IDEAS, AIBL, MR-RATE and future datasets without cohort branches. To add one, see
[Onboarding a dataset](dataset-onboarding.md). List the registered adapters and plugins
with:

```bash
neuroprep-agentic coverage
```
