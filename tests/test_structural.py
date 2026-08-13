from neuroprep_agentic.structural import score_primary_t1, select_primary_t1


def test_primary_t1_selection_prefers_3d_mprage_over_thick_slice_t1() -> None:
    choice = select_primary_t1(
        [
            (
                "acq-coronal",
                {
                    "MRAcquisitionType": "2D",
                    "SeriesDescription": "COR_T1FS",
                    "SliceThickness": 5,
                },
            ),
            (
                "acq-post",
                {
                    "MRAcquisitionType": "2D",
                    "SeriesDescription": "TRA_FL2D_T1_POST",
                    "SliceThickness": 5,
                },
            ),
            (
                "acq-mprage",
                {
                    "MRAcquisitionType": "3D",
                    "SeriesDescription": "SAG_MPRAGE",
                    "SequenceName": "_tfl3d1_16ns",
                    "InversionTime": 0.9,
                    "SliceThickness": 1.1,
                },
            ),
        ]
    )
    assert choice.acquisition_id == "acq-mprage"
    assert choice.score > 0


def test_primary_t1_tie_break_is_deterministic() -> None:
    metadata = {"MRAcquisitionType": "3D", "SliceThickness": 1.0}
    choice = select_primary_t1([("acq-z", metadata), ("acq-a", metadata)])
    assert choice.acquisition_id == "acq-a"
    assert score_primary_t1(metadata)[0] == choice.score
