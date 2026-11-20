"""MRtrix3 diffusion adapter: command planning from BIDS DWI series."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuroprep_agentic.native import mrtrix3_dwi_commands

OUT = Path("/data/out")


def _series(name: str, bvals: list[float], pe: str | None) -> dict:
    base = Path(f"/data/bids/sub-01/dwi/{name}")
    return {
        "nifti": base.with_suffix(".nii.gz"),
        "bval": base.with_suffix(".bval"),
        "bvec": base.with_suffix(".bvec"),
        "json": base.with_suffix(".json"),
        "bvals": bvals,
        "pe": pe,
    }


MAIN = _series("sub-01_acq-AP_dwi", [0] * 6 + [700] * 32 + [1200] * 32 + [2800] * 32, "j-")
REVERSE = _series("sub-01_acq-PA_dwi", [0, 0], "j")


def _names(commands):
    return [command[0] for command in commands]


def test_reverse_phase_b0_enables_topup_pair():
    commands = mrtrix3_dwi_commands([REVERSE, MAIN], OUT, "sub-01", 8)
    assert commands[0][:2] == ["mrconvert", str(MAIN["nifti"])]
    assert "-fslgrad" in commands[0] and "-json_import" in commands[0]
    preproc = next(c for c in commands if c[0] == "dwifslpreproc")
    assert preproc[preproc.index("-se_epi") + 1] == str(OUT / "b0_pair.mif")
    assert "-rpe_header" in preproc and "-align_seepi" in preproc
    first_b0s = next(c for c in commands if "-coord" in c)
    assert first_b0s[first_b0s.index("-coord") + 2] == "0:1"  # as many as the reverse series
    assert _names(commands)[-2:] == ["dwi2tensor", "tensor2metric"]
    assert all("-force" in c and "-nthreads" in c for c in commands)


def test_without_reverse_series_uses_header_only():
    commands = mrtrix3_dwi_commands([MAIN], OUT, "sub-01", 4)
    preproc = next(c for c in commands if c[0] == "dwifslpreproc")
    assert "-se_epi" not in preproc and "mrcat" not in _names(commands)
    same_direction = _series("sub-01_acq-AP2_dwi", [0, 0], "j-")
    commands = mrtrix3_dwi_commands([MAIN, same_direction], OUT, "sub-01", 4)
    assert "mrcat" not in _names(commands)  # not an opposite-phase pair


def test_refuses_without_weighted_volumes_or_phase_encoding():
    with pytest.raises(ValueError, match="diffusion-weighted"):
        mrtrix3_dwi_commands([REVERSE], OUT, "sub-01", 4)
    with pytest.raises(ValueError, match="PhaseEncodingDirection"):
        mrtrix3_dwi_commands([{**MAIN, "pe": None}], OUT, "sub-01", 4)


def test_every_script_scratch_stays_beside_the_outputs():
    # CIAI 419571: dwibiascorrect's default scratch was the in-memory container home and
    # filled it (SIGBUS). Each MRtrix3 script must get an explicit scratch under OUT.
    commands = mrtrix3_dwi_commands([REVERSE, MAIN], OUT, "sub-01", 8)
    for name in ("dwifslpreproc", "dwibiascorrect"):
        command = next(c for c in commands if c[0] == name)
        scratch = command[command.index("-scratch") + 1]
        assert scratch.startswith(str(OUT) + "/")
