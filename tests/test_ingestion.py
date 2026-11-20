from datetime import datetime

from neuroprep_agentic.bids import bids_relative_path
from neuroprep_agentic.identity import IdentityResolver
from neuroprep_agentic.ingestion import IngestionEngine
from neuroprep_agentic.models import ClassificationStatus, DatasetSpec, SourceRecord
from neuroprep_agentic.registry import builtin_registry


def record(
    path: str,
    modality: str,
    description: str,
    session: str = "baseline",
    date: datetime | None = None,
) -> SourceRecord:
    return SourceRecord(
        dataset_id="custom-study",
        source_path=path,
        source_subject_id="source-patient",
        source_session_id=session,
        acquisition_time=date,
        metadata={"Modality": modality, "SeriesDescription": description},
    )


def engine() -> IngestionEngine:
    return IngestionEngine(builtin_registry(), IdentityResolver("a-strong-test-key"))


def test_mixed_modalities_are_consolidated_under_one_patient_session():
    graph = engine().build_graph(
        DatasetSpec(dataset_id="custom-study"),
        [
            record("/raw/T1_MPRAGE.dcm", "MR", "T1w MPRAGE"),
            record("/raw/rest_bold.dcm", "MR", "resting state BOLD"),
            record("/raw/amyloid_pet.dcm", "PT", "florbetapir PET"),
            record("/raw/head_ct.dcm", "CT", "non contrast CT"),
        ],
    )
    assert len(graph.subjects) == 1
    assert len(graph.subjects[0].sessions) == 1
    assert {item.selected_plugin_id for item in graph.acquisitions} == {
        "mri.t1w",
        "mri.bold",
        "pet",
        "ct",
    }


def test_flair_is_not_mislabeled_as_t2w():
    graph = engine().build_graph(
        DatasetSpec(dataset_id="custom-study"),
        [record("/raw/T2_FLAIR.dcm", "MR", "T2 FLAIR dark fluid")],
    )
    acquisition = graph.acquisitions[0]
    assert acquisition.status is ClassificationStatus.RECOGNIZED
    assert acquisition.selected_plugin_id == "mri.flair"


def test_anat2_directory_does_not_exclude_t1w():
    graph = engine().build_graph(
        DatasetSpec(dataset_id="custom-study"),
        [record("/raw/anat2/sub-01_T1w.nii.gz", "MR", "MPRAGE T1w")],
    )
    acquisition = graph.acquisitions[0]
    assert acquisition.status is ClassificationStatus.RECOGNIZED
    assert acquisition.selected_plugin_id == "mri.t1w"


def test_explicit_t1w_suffix_beats_generic_gre_and_tse_metadata():
    records = []
    for run, sequence in (("01", "GRE"), ("02", "TSE")):
        item = record(f"/raw/sub-01_run-{run}_T1w.nii.gz", "MR", sequence)
        item.metadata["SequenceName"] = sequence
        records.append(item)
    graph = engine().build_graph(DatasetSpec(dataset_id="custom-study"), records)
    assert [item.selected_plugin_id for item in graph.acquisitions] == ["mri.t1w", "mri.t1w"]
    assert all(item.status is ClassificationStatus.RECOGNIZED for item in graph.acquisitions)


def test_minip_swi_and_tof_angio_are_recognized():
    minip = record("/raw/sub-01_minIP.nii.gz", "MR", "mIP_Images(SW)")
    minip.metadata["SequenceName"] = "_swi3d1r"
    tof = record("/raw/sub-01_angio.nii.gz", "MR", "TOF 3D multi slab non con")
    tof.metadata["SequenceVariant"] = "TOF_SP"
    graph = engine().build_graph(DatasetSpec(dataset_id="custom-study"), [minip, tof])
    assert [item.selected_plugin_id for item in graph.acquisitions] == [
        "mri.swi",
        "mri.angio",
    ]
    assert graph.acquisitions[0].bids is not None
    assert graph.acquisitions[0].bids.suffix == "T2starw"
    assert str(bids_relative_path(graph.acquisitions[0])).endswith("_T2starw.nii.gz")
    assert all(item.status is ClassificationStatus.RECOGNIZED for item in graph.acquisitions)


def test_siemens_fieldmap_collection_maps_to_bids_phase_difference_case():
    items = []
    for path, echo, image_type in (
        ("/raw/echo-1_fieldmap.nii.gz", 0.00492, ["M"]),
        ("/raw/echo-2_fieldmap.nii.gz", 0.00738, ["M"]),
        ("/raw/run-02_fieldmap.nii.gz", 0.00738, ["P"]),
    ):
        item = record(path, "MR", "Field Mapping")
        item.metadata.update({"EchoTime": echo, "ImageType": image_type})
        items.append(item)
    graph = engine().build_graph(DatasetSpec(dataset_id="custom-study"), items)
    assert [item.bids.suffix for item in graph.acquisitions if item.bids] == [
        "magnitude1",
        "magnitude2",
        "phasediff",
    ]
    phasediff = graph.acquisitions[-1]
    assert phasediff.source.metadata["EchoTime1"] == 0.00492
    assert phasediff.source.metadata["EchoTime2"] == 0.00738


def test_unknown_mr_is_quarantined_instead_of_guessed():
    graph = engine().build_graph(
        DatasetSpec(dataset_id="custom-study"),
        [record("/raw/localizer.dcm", "MR", "unknown localizer")],
    )
    assert graph.acquisitions[0].status is ClassificationStatus.AMBIGUOUS
    assert len(graph.quarantined) == 1


def test_unsupported_input_is_preserved_and_explicit():
    graph = engine().build_graph(
        DatasetSpec(dataset_id="custom-study"),
        [record("/raw/proprietary.xyz", "OTHER", "unknown acquisition")],
    )
    assert graph.acquisitions[0].status is ClassificationStatus.UNSUPPORTED
    assert len(graph.unsupported) == 1
    assert graph.acquisitions[0].source.source_path == "/raw/proprietary.xyz"


def test_bids_path_uses_pseudonymous_subject_and_session():
    graph = engine().build_graph(
        DatasetSpec(dataset_id="custom-study"),
        [record("/raw/T1_MPRAGE.dcm", "MR", "T1w MPRAGE")],
    )
    acquisition = graph.acquisitions[0]
    path = str(bids_relative_path(acquisition))
    assert acquisition.source.source_subject_id not in path
    assert path.startswith(f"{acquisition.subject_id}/{acquisition.session_id}/anat/")
    assert path.endswith("_T1w.nii.gz")


def test_repeated_suffixes_have_collision_safe_bids_paths():
    graph = engine().build_graph(
        DatasetSpec(dataset_id="custom-study"),
        [
            record("/raw/T1_repeat_1.dcm", "MR", "T1w MPRAGE"),
            record("/raw/T1_repeat_2.dcm", "MR", "T1w MPRAGE"),
        ],
    )
    paths = [str(bids_relative_path(acquisition)) for acquisition in graph.acquisitions]
    assert len(paths) == len(set(paths)) == 2
    assert all("_acq-" in path for path in paths)
    assert all(
        acquisition.source.source_subject_id not in path
        for acquisition, path in zip(graph.acquisitions, paths, strict=True)
    )


def test_pet_paths_use_bids_tracer_and_run_entities_instead_of_acquisition():
    records = [
        record(f"/raw/av45_repeat_{index}_pet.nii.gz", "PT", "AV45 amyloid PET")
        for index in (1, 2)
    ]
    for item in records:
        item.metadata["TracerName"] = "AV-45"
    graph = engine().build_graph(DatasetSpec(dataset_id="custom-study"), records)
    paths = [str(bids_relative_path(acquisition)) for acquisition in graph.acquisitions]
    assert len(paths) == len(set(paths)) == 2
    assert all("_trc-AV45_run-" in path for path in paths)
    assert all("_acq-" not in path for path in paths)


def test_dates_create_longitudinal_sessions_when_source_labels_are_absent():
    records = [
        record("/raw/v1_T1.dcm", "MR", "T1w MPRAGE", session=None, date=datetime(2025, 1, 1)),
        record("/raw/v2_T1.dcm", "MR", "T1w MPRAGE", session=None, date=datetime(2026, 1, 1)),
    ]
    for item in records:
        item.source_session_id = None
    graph = engine().build_graph(DatasetSpec(dataset_id="custom-study"), records)
    assert [session.session_id for session in graph.subjects[0].sessions] == [
        "ses-d00000",
        "ses-d00365",
    ]


def test_reverse_phase_b0_only_dwi_is_staged_as_an_epi_fieldmap(tmp_path):
    # CIAI 422645: QSIPrep processed the 2-volume reverse-PE b0 series as a DWI run of its
    # own and Eddy failed on its all-zero gradients. BIDS stages it as fmap/*_epi.
    def dwi(name: str, bvals: str, direction: str) -> SourceRecord:
        path = tmp_path / f"{name}_dwi.nii.gz"
        path.write_bytes(b"")
        (tmp_path / f"{name}_dwi.bval").write_text(bvals + "\n")
        return SourceRecord(
            dataset_id="custom-study",
            source_path=str(path),
            source_subject_id="source-patient",
            source_session_id="preop",
            metadata={
                "Modality": "MR",
                "SeriesDescription": "diffusion dwi",
                "PhaseEncodingDirection": direction,
            },
        )

    graph = engine().build_graph(
        DatasetSpec(dataset_id="custom-study"),
        [dwi("main", "0 1000 1000 0", "j-"), dwi("reverse", "0 0", "j"), dwi("b0", "0", "j-")],
    )
    by_name = {
        item.source.source_path.rsplit("/", 1)[1]: item for item in graph.acquisitions
    }
    main = by_name["main_dwi.nii.gz"]
    reverse = by_name["reverse_dwi.nii.gz"]
    same_direction = by_name["b0_dwi.nii.gz"]

    path = bids_relative_path(reverse)
    assert path.parent.name == "fmap" and path.name.endswith("_dir-j_epi.nii.gz")
    intended = reverse.bids.sidecar["IntendedFor"]
    assert intended == [str(bids_relative_path(main).relative_to(main.subject_id))]
    assert intended[0].startswith("ses-") and intended[0].endswith("_dwi.nii.gz")
    # No opposite-PE partner: a b0-only series in the main direction stays a DWI run.
    assert bids_relative_path(same_direction).parent.name == "dwi"
    assert bids_relative_path(main).parent.name == "dwi"
