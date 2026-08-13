from __future__ import annotations

import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

import neuroprep_agentic.native as native
from neuroprep_agentic.execution_models import ExecutableTask, ExecutionScope


def _task(
    root: Path,
    *,
    executor_id: str,
    acquisition_ids: list[str],
    inputs: list[Path],
    source_metadata: dict[str, dict] | None = None,
) -> ExecutableTask:
    return ExecutableTask(
        task_id=f"{executor_id}-test",
        logical_task_ids=["logical"],
        executor_id=executor_id,
        plugin_ids=["test"],
        stages=["test"],
        scope=ExecutionScope.SESSION,
        subject_id="sub-01",
        session_id="ses-01",
        acquisition_ids=acquisition_ids,
        command=["true"],
        input_paths=list(map(str, inputs)),
        output_root=str(root / "derivatives"),
        work_dir=str(root / "work"),
        resource_class="cpu",
        cpus=1,
        memory_gb=4,
        gpus=0,
        walltime_minutes=10,
        fingerprint="test",
        metadata={
            "bids_inputs": list(map(str, inputs)),
            "source_metadata": source_metadata or {},
        },
    )


def test_vendor_mrs_staging_uses_spec2nii_and_anonymizes(monkeypatch, nfs_tmp: Path) -> None:
    source = nfs_tmp / "source.dat"
    source.write_bytes(b"vendor-mrs")
    target = nfs_tmp / "bids" / "sub-01_mrs.nii.gz"
    task = _task(
        nfs_tmp,
        executor_id="neuroprep-bids-stage",
        acquisition_ids=["mrs-metab"],
        inputs=[source],
    )
    task.metadata.update(
        {
            "source_path": str(source),
            "bids_path": str(target),
            "selected_plugin_id": "mri.mrs",
            "source_metadata": {
                "ResonantNucleus": ["1H"],
                "SpectrometerFrequency": [123.2],
            },
            "bids_sidecar": {},
        }
    )
    observed: list[str] = []

    def fake_run(command, **kwargs):
        observed.extend(command)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"nifti-mrs")
        target.with_name("sub-01_mrs.json").write_text("{}")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(native.subprocess, "run", fake_run)
    result = native.stage_bids(task)
    assert result["converter"] == "spec2nii 0.8.15"
    assert "auto" in observed
    assert "--anon" in observed
    assert "--override_nucleus" in observed
    sidecar = json.loads(target.with_name("sub-01_mrs.json").read_text())
    assert sidecar["ConversionSoftware"] == "spec2nii"


def test_fsl_mrs_wrapper_requires_water_and_emits_concentrations(
    monkeypatch, nfs_tmp: Path
) -> None:
    metabolite = nfs_tmp / "metab.nii.gz"
    water = nfs_tmp / "water.nii.gz"
    basis = nfs_tmp / "basis"
    metabolite.write_bytes(b"metab")
    water.write_bytes(b"water")
    basis.mkdir()
    metadata = {
        "metab": {
            "MRSRole": "metabolite",
            "BasisPath": str(basis),
            "ResonantNucleus": ["1H"],
            "SpectrometerFrequency": [123.2],
        },
        "water": {
            "MRSRole": "water_reference",
            "BasisPath": str(basis),
            "ResonantNucleus": ["1H"],
            "SpectrometerFrequency": [123.2],
        },
    }
    task = _task(
        nfs_tmp,
        executor_id="fsl-mrs",
        acquisition_ids=["metab", "water"],
        inputs=[metabolite, water],
        source_metadata=metadata,
    )
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        if command[0].endswith("fsl_mrs_preproc"):
            (output / "metab.nii.gz").write_bytes(b"processed")
            (output / "wref.nii.gz").write_bytes(b"water")
        else:
            (output / "concentrations.csv").write_text("metabolite,mean\nNAA,1.0\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(native.subprocess, "run", fake_run)
    result = native._mrs(task, nfs_tmp / "mrs-output")
    assert result["status"] == "pass"
    assert len(commands) == 2
    assert "--reference" in commands[0]
    assert "--basis" in commands[1]
    assert "--h2o" in commands[1]
    assert result["concentration_tables"]


def test_spect_preprocessing_validates_and_canonicalizes_nifti(nfs_tmp: Path) -> None:
    source = nfs_tmp / "spect.nii.gz"
    data = np.arange(27, dtype=np.float32).reshape((3, 3, 3))
    nib.save(nib.Nifti1Image(data, np.diag([-2.0, 2.0, 2.0, 1.0])), source)
    task = _task(nfs_tmp, executor_id="neuroprep-spect", acquisition_ids=["spect"], inputs=[source])
    result = native._spect(task, nfs_tmp / "spect-output")
    assert result["status"] == "pass"
    target = Path(result["images"][0]["target"])
    assert target.is_file()
    assert nib.aff2axcodes(nib.load(target).affine)[0] == "R"
    assert result["images"][0]["intensity_units_preserved"] is True


def test_angio_preprocessing_validates_and_preserves_native_intensities(nfs_tmp: Path) -> None:
    source = nfs_tmp / "tof_angio.nii.gz"
    data = np.arange(27, dtype=np.float32).reshape((3, 3, 3))
    nib.save(nib.Nifti1Image(data, np.diag([-2.0, 2.0, 2.0, 1.0])), source)
    task = _task(
        nfs_tmp,
        executor_id="neuroprep-angio",
        acquisition_ids=["angio"],
        inputs=[source],
    )
    result = native._angio(task, nfs_tmp / "angio-output")
    assert result["status"] == "pass"
    target = Path(result["images"][0]["target"])
    assert target.is_file()
    assert nib.aff2axcodes(nib.load(target).affine)[0] == "R"
    assert result["images"][0]["intensity_units_preserved"] is True


def test_ultrasound_video_is_losslessly_normalized(monkeypatch, nfs_tmp: Path) -> None:
    source = nfs_tmp / "doppler.mp4"
    source.write_bytes(b"video")
    task = _task(
        nfs_tmp,
        executor_id="neuroprep-ultrasound",
        acquisition_ids=["us"],
        inputs=[source],
    )
    monkeypatch.setattr(native.shutil, "which", lambda name: f"/tools/{name}")

    def fake_run(command, **kwargs):
        if command[0].endswith("ffprobe"):
            payload = {"streams": [{"width": 640, "height": 480, "avg_frame_rate": "30/1"}]}
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        Path(command[-1]).write_bytes(b"ffv1")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(native.subprocess, "run", fake_run)
    result = native._ultrasound(task, nfs_tmp / "ultrasound-output")
    assert result["status"] == "pass"
    assert result["videos"][0]["normalization"].startswith("lossless FFV1")
    assert Path(result["videos"][0]["target"]).is_file()


def test_behavior_json_is_normalized_to_tsv(nfs_tmp: Path) -> None:
    source = nfs_tmp / "assessment.json"
    source.write_text(json.dumps([{"score": 12, "visit": "baseline"}]))
    task = _task(
        nfs_tmp, executor_id="neuroprep-behavior", acquisition_ids=["beh"], inputs=[source]
    )
    result = native._tabular(task, nfs_tmp / "behavior-output", "behavior")
    assert result["status"] == "pass"
    assert result["tables"][0]["rows"] == 1
    assert Path(result["tables"][0]["target"]).read_text().startswith("score\tvisit")


def test_physiology_edf_is_converted_to_fif(monkeypatch, nfs_tmp: Path) -> None:
    import mne

    source = nfs_tmp / "physio.edf"
    source.write_bytes(b"edf")
    task = _task(
        nfs_tmp, executor_id="neuroprep-physiology", acquisition_ids=["phys"], inputs=[source]
    )

    class FakeRaw:
        ch_names = ["ECG", "RESP"]
        n_times = 100
        info = {"sfreq": 50.0}

        def save(self, target, **_kwargs):
            Path(target).write_bytes(b"fif")

    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *_args, **_kwargs: FakeRaw())
    result = native._tabular(task, nfs_tmp / "physiology-output", "physiology")
    assert result["status"] == "pass"
    assert result["signals"][0]["channels"] == 2
    assert Path(result["signals"][0]["target"]).is_file()


def test_sensor_features_compute_bandpower(nfs_tmp: Path) -> None:
    import mne
    import pandas as pd

    source = nfs_tmp / "rest_raw.fif"
    sampling_frequency = 128.0
    times = np.arange(512) / sampling_frequency
    data = np.vstack([np.sin(2 * np.pi * 10 * times), np.sin(2 * np.pi * 6 * times)])
    raw = mne.io.RawArray(
        data, mne.create_info(["C3", "C4"], sampling_frequency, ch_types="eeg"), verbose="error"
    )
    raw.save(source, overwrite=True, verbose="error")
    task = _task(
        nfs_tmp, executor_id="neuroprep-eeg-features", acquisition_ids=["eeg"], inputs=[source]
    )
    result = native._sensor_features(task, nfs_tmp / "feature-output")
    target = Path(result["features"][0]["target"])
    frame = pd.read_csv(target, sep="\t")
    assert list(frame["channel"]) == ["C3", "C4"]
    assert frame.loc[0, "alpha_power"] > frame.loc[0, "theta_power"]
    assert frame.loc[1, "theta_power"] > frame.loc[1, "alpha_power"]


def test_behavior_json_staging_produces_real_tsv(nfs_tmp: Path) -> None:
    source = nfs_tmp / "assessment-source.json"
    source.write_text(json.dumps([{"score": 12, "visit": "baseline"}]))
    target = nfs_tmp / "bids" / "sub-01_beh.tsv"
    task = _task(
        nfs_tmp, executor_id="neuroprep-bids-stage", acquisition_ids=["beh"], inputs=[source]
    )
    task.metadata.update(
        {
            "source_path": str(source),
            "bids_path": str(target),
            "selected_plugin_id": "behavior",
            "source_metadata": {},
            "bids_sidecar": {},
        }
    )
    result = native.stage_bids(task)
    assert result["rows"] == 1
    assert target.read_text().startswith("score\tvisit")
    assert json.loads(target.with_suffix(".json").read_text())["ConversionSoftware"] == "pandas"


def test_asl_staging_derives_evidence_backed_context(nfs_tmp: Path) -> None:
    source = nfs_tmp / "source_asl.nii.gz"
    data = np.zeros((3, 3, 3, 5), dtype=np.float32)
    data[..., 0] = 100
    data[..., 1] = 10
    data[..., 2] = 20
    data[..., 3] = 11
    data[..., 4] = 21
    nib.save(nib.Nifti1Image(data, np.eye(4)), source)
    target = nfs_tmp / "bids" / "sub-01_asl.nii.gz"
    task = _task(
        nfs_tmp, executor_id="neuroprep-bids-stage", acquisition_ids=["asl"], inputs=[source]
    )
    task.metadata.update(
        {
            "source_path": str(source),
            "bids_path": str(target),
            "selected_plugin_id": "mri.asl",
            "source_metadata": {"M0Type": "Included", "ArterialSpinLabelingType": "PASL"},
            "bids_sidecar": {},
        }
    )
    result = native.stage_bids(task)
    context = Path(result["asl_context"]["path"])
    assert result["asl_context"]["pairs"] == 2
    assert context.read_text().splitlines() == [
        "volume_type",
        "m0scan",
        "label",
        "control",
        "label",
        "control",
    ]
    sidecar = json.loads(target.with_name("sub-01_asl.json").read_text())
    assert sidecar["M0Type"] == "Included"
    assert sidecar["TotalAcquiredPairs"] == 2


def test_bids_init_and_validator_wrapper(monkeypatch, nfs_tmp: Path) -> None:
    bids_root = nfs_tmp / "bids"
    init = _task(
        nfs_tmp, executor_id="neuroprep-bids-init", acquisition_ids=[], inputs=[]
    ).model_copy(
        update={
            "output_root": str(bids_root),
            "metadata": {"dataset_id": "unit-test", "bids_version": "1.11.1"},
        }
    )
    initialized = native._bids_init(init)
    description = json.loads(Path(initialized["dataset_description"]).read_text())
    assert description["DatasetType"] == "raw"
    assert "**/x-ultrasound/**" in Path(initialized["bidsignore"]).read_text()
    assert "**/ct/" in Path(initialized["bidsignore"]).read_text()
    (bids_root / "sub-01" / "ses-01" / "ct").mkdir(parents=True)

    image = nfs_tmp / "bids-validator.sif"
    image.write_bytes(b"sif")
    validation = _task(nfs_tmp, executor_id="bids-validator", acquisition_ids=[], inputs=[])
    validation.metadata.update({"bids_root": str(bids_root), "container_image": str(image)})

    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        report = Path(validation.output_root) / "bids-validation" / "bids-validator.json"
        report.write_text(json.dumps({"summary": {"errors": 0}}))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(native.subprocess, "run", fake_run)
    result = native._bids_validate(validation)
    assert result["status"] == "pass"
    assert result["summary"]["errors"] == 0
    assert "--config" in commands[0]
    config = json.loads(Path(result["validator_config"]).read_text())
    assert config["ignore"] == [
        {"code": "NOT_INCLUDED", "location": "/sub-01/ses-01/ct/"}
    ]


def test_multimodal_registration_uses_t1_and_3d_dwi_reference(
    monkeypatch, nfs_tmp: Path
) -> None:
    fieldmap = nfs_tmp / "sub-01_magnitude1.nii.gz"
    dwi = nfs_tmp / "sub-01_dwi.nii.gz"
    t1w = nfs_tmp / "sub-01_T1w.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), np.eye(4)), fieldmap)
    dwi_data = np.stack(
        [
            np.full((3, 3, 3), 10, dtype=np.float32),
            np.full((3, 3, 3), 100, dtype=np.float32),
            np.full((3, 3, 3), 30, dtype=np.float32),
        ],
        axis=3,
    )
    nib.save(nib.Nifti1Image(dwi_data, np.eye(4)), dwi)
    dwi.with_name("sub-01_dwi.bval").write_text("0 1000 0\n")
    nib.save(nib.Nifti1Image(np.full((3, 3, 3), 5, dtype=np.float32), np.eye(4)), t1w)
    task = _task(
        nfs_tmp,
        executor_id="neuroprep-multimodal-registration",
        acquisition_ids=["fmap", "dwi", "t1w"],
        inputs=[fieldmap, dwi, t1w],
    )
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        prefix = Path(command[command.index("-o") + 1])
        prefix.with_name(prefix.name + "0GenericAffine.mat").write_bytes(b"transform")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(native.subprocess, "run", fake_run)
    result = native._ants_registration(task, nfs_tmp / "registration", longitudinal=False)
    assert len(commands) == 1
    assert Path(commands[0][commands[0].index("-f") + 1]) == t1w
    moving = Path(commands[0][commands[0].index("-m") + 1])
    assert nib.load(moving).shape == (3, 3, 3)
    assert np.allclose(nib.load(moving).get_fdata(), 20)
    assert all("magnitude1" not in item["source"] for item in result["references"])
    assert result["references"][1]["strategy"] == "b0mean"
    assert result["transforms"]
