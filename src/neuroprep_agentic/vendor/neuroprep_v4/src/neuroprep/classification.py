from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from .models import Modality


_SUFFIX_MAP = {
    "t1w": Modality.T1W,
    "unit1": Modality.T1W,
    "t2w": Modality.T2W,
    "flair": Modality.FLAIR,
    "bold": Modality.FMRI,
    "sbref": Modality.FMRI,
    "dwi": Modality.DWI,
    "dti": Modality.DWI,
    "pet": Modality.PET,
    "ct": Modality.CT,
    "swi": Modality.SWI,
    "t2star": Modality.SWI,
    "magnitude": Modality.SWI,
    "phase": Modality.SWI,
    "angio": Modality.MRA,
    "mra": Modality.MRA,
    "tof": Modality.MRA,
    "asl": Modality.ASL,
    "m0scan": Modality.ASL,
    "eeg": Modality.EEG,
    "meg": Modality.MEG,
    "ieeg": Modality.IEEG,
    "fieldmap": Modality.FIELDMAP,
    "phasediff": Modality.FIELDMAP,
    "epi": Modality.FIELDMAP,
    "pd": Modality.PD,
    "pdw": Modality.PD,
}

_KEYWORD_RULES: list[tuple[Modality, tuple[str, ...]]] = [
    (Modality.FIELDMAP, ("fieldmap", "field map", "field mapping", "phasediff")),
    (Modality.FLAIR, ("flair", "t2 flair", "dark fluid")),
    (
        Modality.SWI,
        (
            "swi",
            "qsm",
            "susceptibility",
            "swan",
            "venobold",
            "multi echo gre",
            "gre 3d 5echo",
            "t2*",
            "t2star",
        ),
    ),
    (Modality.DWI, ("dwi", "diffusion", "dti", "tracew", "ep2d_diff")),
    (Modality.FMRI, ("bold", "fmri", "resting state", "resting-state", "task-")),
    (Modality.ASL, ("asl", "arterial spin", "pcasl", "pasl")),
    (
        Modality.PET,
        (
            "pet",
            "fdg",
            "amyloid",
            "pib",
            "florbet",
            "flortaucipir",
            "pi2620",
            "mk6240",
            "av1451",
            "amyvid",
        ),
    ),
    (Modality.MRA, ("mra", "mr angi", "time of flight", "tof", "artery", "angiography")),
    (Modality.CT, (" ct ", "computed tomography", "head ct", "brain ct")),
    (Modality.T1W, ("t1", "mprage", " mpr ", "spgr", "bravo", "tfl3d")),
    (Modality.T2W, ("t2", "space", "cube", "propeller")),
    (Modality.PD, ("proton density", " pd ")),
]

_LOCALIZER = re.compile(r"locali[sz]er|scout|survey|screen.?save|dose report", re.I)


def classify_suffix(suffix: str | None) -> Modality:
    return _SUFFIX_MAP.get((suffix or "").lower(), Modality.UNKNOWN)


def classify_series(path: Path, metadata: Mapping[str, Any] | None = None) -> Modality:
    """Classify a series using BIDS suffixes, DICOM fields, then description rules.

    The hierarchy follows the robust design pattern used by MR-RATE while using
    independently implemented rules and a broader modality vocabulary.
    """

    metadata = metadata or {}
    # BIDS suffixes are authoritative for NIfTI/JSON files. Archive filenames
    # often end in scanner or reconstruction tokens (for example PET "_TOF"),
    # so treating those tokens as BIDS suffixes creates false positives.
    is_bids_like_file = path.name.lower().endswith(
        (
            ".nii",
            ".nii.gz",
            ".json",
            ".edf",
            ".bdf",
            ".vhdr",
            ".set",
            ".fif",
            ".con",
            ".sqd",
            ".nwb",
            ".mefd",
            ".ds",
        )
    )
    if is_bids_like_file:
        suffix = infer_suffix(path)
        modality = classify_suffix(suffix)
        if modality is not Modality.UNKNOWN:
            return modality

    text_fields = (
        path.name,
        str(metadata.get("SeriesDescription", "")),
        str(metadata.get("ProtocolName", "")),
        str(metadata.get("SequenceName", "")),
        str(metadata.get("PulseSequenceName", "")),
    )
    normalized = " ".join(text_fields).lower().replace("_", " ").replace("-", " ")
    if _LOCALIZER.search(normalized):
        return Modality.UNKNOWN

    image_type = str(metadata.get("ImageType", "")).upper()
    diffusion_b = metadata.get("DiffusionBValue")
    if diffusion_b not in (None, "", 0, 0.0) or "DIFFUSION" in image_type:
        return Modality.DWI

    units = str(metadata.get("Units", "")).upper()
    tracer = metadata.get("TracerName") or metadata.get("Radiopharmaceutical")
    if tracer or units in {"BQML", "CNTS", "GML"}:
        return Modality.PET

    modality_tag = str(metadata.get("Modality", "")).upper()
    if modality_tag == "CT":
        return Modality.CT
    if modality_tag in {"PT", "PET"}:
        return Modality.PET

    padded = f" {normalized} "
    for candidate, keywords in _KEYWORD_RULES:
        if any(keyword in padded for keyword in keywords):
            return candidate
    return Modality.UNKNOWN


def infer_suffix(path: Path) -> str | None:
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    else:
        name = Path(name).stem
    token = name.rsplit("_", 1)[-1]
    return token if token else None
