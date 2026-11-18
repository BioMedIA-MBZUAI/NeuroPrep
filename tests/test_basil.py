"""BIDS-ASL metadata -> FSL oxford_asl flags (BASIL adapter)."""

from __future__ import annotations

import pytest

from neuroprep_agentic.native import oxford_asl_arguments

PCASL = {
    "ArterialSpinLabelingType": "PCASL",
    "PostLabelingDelay": 1.5,
    "LabelingDuration": 1.6,
    "RepetitionTime": 3.5,
    "MRAcquisitionType": "3D",
}


def test_pcasl_with_included_m0_matches_ds000240_layout():
    types = ["m0scan"] * 10 + ["label", "control"] * 50
    flags, data, m0 = oxford_asl_arguments(PCASL, types)
    assert flags == [
        "--iaf", "tc", "--casl", "--plds", "1.5", "--bolus", "1.6",
        "--tr", "3.5", "--cmethod", "voxel",
    ]
    assert data == list(range(10, 110)) and m0 == list(range(10))


def test_pasl_delay_is_already_the_inversion_time():
    meta = {
        "ArterialSpinLabelingType": "PASL",
        "PostLabelingDelay": 1.8,
        "BolusCutOffDelayTime": 0.7,
        "RepetitionTimePreparation": 4.0,
        "RepetitionTime": 3.0,
    }
    flags, _, _ = oxford_asl_arguments(meta, ["m0scan", "control", "label"])
    assert flags[:2] == ["--iaf", "ct"]
    assert "--casl" not in flags
    assert flags[flags.index("--tis") + 1] == "1.8" and flags[flags.index("--bolus") + 1] == "0.7"
    assert flags[flags.index("--tr") + 1] == "4"
    with pytest.raises(ValueError, match="BolusCutOffDelayTime"):
        oxford_asl_arguments({**meta, "BolusCutOffDelayTime": None}, ["m0scan", "label", "control"])


def test_multi_pld_order_and_2d_slice_timing():
    types = ["m0scan"] + ["label", "control"] * 6
    cyclic = [0] + [p for p in (0.5, 1.0, 1.5) * 2 for _ in range(2)]
    flags, _, _ = oxford_asl_arguments(
        {**PCASL, "PostLabelingDelay": cyclic, "MRAcquisitionType": "2D",
         "SliceTiming": [0.0, 0.04, 0.08, 0.12]},
        types,
    )
    assert flags[flags.index("--ibf") + 1] == "rpt"
    assert flags[flags.index("--plds") + 1] == "0.5,1,1.5"
    assert flags[flags.index("--slicedt") + 1] == "0.04"
    grouped = [0] + [p for p in (0.5, 0.5, 1.0, 1.0, 1.5, 1.5) for _ in range(2)]
    flags, _, _ = oxford_asl_arguments({**PCASL, "PostLabelingDelay": grouped}, types)
    assert flags[flags.index("--ibf") + 1] == "tis"


def test_refuses_without_m0_or_known_labeling():
    with pytest.raises(ValueError, match="M0"):
        oxford_asl_arguments(PCASL, ["label", "control"])
    with pytest.raises(ValueError, match="ArterialSpinLabelingType"):
        oxford_asl_arguments({**PCASL, "ArterialSpinLabelingType": "VSASL"}, ["m0scan", "label"])


def test_staging_keeps_source_aslcontext_with_several_m0_volumes(tmp_path) -> None:
    import nibabel as nib
    import numpy as np

    from neuroprep_agentic.native import _copy_asl_context

    types = ["m0scan"] * 3 + ["control", "label"] * 4
    source = tmp_path / "src" / "sub-01_asl.nii.gz"
    source.parent.mkdir()
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, len(types))), np.eye(4)), source)
    (source.parent / "sub-01_aslcontext.tsv").write_text("volume_type\n" + "\n".join(types) + "\n")
    target = tmp_path / "bids" / "sub-x_acq-y_asl.nii.gz"
    target.parent.mkdir()
    target.write_bytes(source.read_bytes())

    context = _copy_asl_context(source, target)

    assert context["pairs"] == 4 and context["volumes"] == 11
    assert (target.parent / "sub-x_acq-y_aslcontext.tsv").read_text().split()[1:] == types
