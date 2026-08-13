# Modality execution matrix

This matrix describes routing, not cohort qualification. A dataset is released
only after preflight and real-data artifact/QC validation pass.

| Family | Inputs | BIDS target | Deterministic executor |
|---|---|---|---|
| Structural MRI | T1w | anat/T1w | SmriPrep 0.19.1 canonical derivatives plus DeepPrep 25.1.0 segmentation |
| Structural MRI | T2w, FLAIR, PDw | anat | NeuroPrep-v4 spatial pipeline |
| MR angiography | TOF and other MRA | anat/angio | Native-intensity-preserving canonical NIfTI validation and angiography QC |
| Quantitative MRI | UNIT1, qMRI maps, T2-star | anat | Unit-preserving native NIfTI validation |
| Functional MRI | BOLD task/rest | func/bold | fMRIPrep 25.2.5 plus XCP-D 26.1.1 |
| Perfusion MRI | ASL | perf/asl | ASLPrep 26.0.3 |
| Diffusion MRI | DWI, DTI, DKI, HARDI | dwi | QSIPrep 26.0.0 plus QSIRecon 1.1.1 DSI Studio GQI |
| Susceptibility MRI | SWI, magnitude/phase QSM | anat | NeuroPrep-v4 plus QSMxT routing |
| Field maps | phasediff, magnitude, EPI field maps | fmap | Metadata and IntendedFor validator |
| Spectroscopy | SVS/MRSI and vendor raw | mrs | spec2nii 0.8.15 plus FSL-MRS 2.4.17 with basis/water metadata gate |
| PET | static/dynamic PET | pet | PETPrep 0.0.8 |
| CT | noncontrast, CTA, perfusion CT | provisional ct | HU-preserving NeuroPrep-v4 spatial path |
| SPECT | nuclear medicine SPECT | provisional x-nuclear | Quantitative-preserving canonical NIfTI validation and QC |
| EEG | EDF, BDF, BrainVision, EEGLAB, FIF | eeg | NeuroPrep-v4 MNE pipeline |
| iEEG | SEEG/ECoG | ieeg | NeuroPrep-v4 MNE pipeline |
| MEG | FIF, CTF, KIT | meg | NeuroPrep-v4 MNE pipeline |
| fNIRS | SNIRF | nirs | MNE optical-density/SCI/Beer-Lambert path |
| Microscopy | TIFF, OME-TIFF | micr | Native validation and denoising path |
| Physiology | cardiac, respiratory, ECG tables | func/physio | Typed tabular normalization |
| Behavior | CSV/TSV assessments | beh | Typed longitudinal table normalization |
| Genetics | VCF, BCF, TSV | genetic extension | Format validation and immutable copy |
| Ultrasound | NIfTI neurosonography, MP4 Doppler | provisional ultrasound | Canonical NIfTI QC or lossless FFV1 video normalization |

## Qualification states

- Registered: detector, BIDS mapping, and logical tasks exist.
- Executable: every logical route resolves to a command and artifact contract.
- Installed: all required images, tools, licenses, and metadata pass preflight.
- Dataset-qualified: representative data pass artifact and scientific QC checks.
- Released: every required task is complete and the atomic indexes validate.

Provisional BIDS-extension families are intentionally not described as
scientifically harmonized until a modality expert approves a stronger executor
and validation contract.
