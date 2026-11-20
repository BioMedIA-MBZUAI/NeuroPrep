from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest
from pydantic import ValidationError

from neuroprep_agentic import interface


def test_interface_request_forbids_operational_fields() -> None:
    with pytest.raises(ValidationError, match="command"):
        interface.InterfacePlanRequest.model_validate(
            {
                "request": "Prepare structural MRI.",
                "source_root": "/approved/source",
                "output_root": "/approved/output",
                "command": "unsafe",
            }
        )


def test_compile_interface_plan_reuses_guarded_prompt_run(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path
) -> None:
    source = nfs_tmp / "source"
    source.mkdir()
    output = nfs_tmp / "run"
    output.mkdir()
    (output / "policy.json").write_text('{"objective": "standard"}', encoding="utf-8")
    (output / "backend-plan.json").write_text('{"ready": true}', encoding="utf-8")
    (output / "preflight.json").write_text('{"ready": true}', encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_prepare(request: str, **kwargs):
        observed["request"] = request
        observed.update(kwargs)
        return {
            "ready": True,
            "binding": {"source_root": str(source), "output_root": str(output)},
        }

    monkeypatch.setattr(interface, "prepare_prompt_run", fake_prepare)

    result = interface.compile_interface_plan(
        {
            "request": "Prepare structural MRI.",
            "source_root": str(source),
            "output_root": str(output),
            "openjev_review": True,
        },
        agentic_project=str(nfs_tmp / "project"),
    )

    assert result["policy"]["objective"] == "standard"
    assert result["backend_plan"]["ready"] is True
    assert result["preflight"]["ready"] is True
    assert observed["openjev_review"] is True
    assert observed["agentic_project"] == str(nfs_tmp / "project")
    assert "initialize" not in observed
    assert "submit" not in observed


def test_interface_reports_missing_source_without_pydantic_trace(nfs_tmp: Path) -> None:
    with pytest.raises(ValueError, match=r"^Source folder does not exist: ") as caught:
        interface.compile_interface_plan(
            {
                "request": "Prepare structural MRI.",
                "source_root": str(nfs_tmp / "missing"),
                "output_root": str(nfs_tmp / "output"),
            }
        )

    assert "pydantic.dev" not in str(caught.value)


def test_public_validation_error_is_human_readable() -> None:
    with pytest.raises(ValidationError) as caught:
        interface.InterfacePlanRequest.model_validate({"request": ""})

    message = interface._public_error(caught.value)

    assert "pydantic.dev" not in message
    assert "request:" in message


def test_interface_serves_assets_and_plan_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        interface,
        "compile_interface_plan",
        lambda payload, agentic_project=None: {
            "result": {"ready": True, "request": payload["request"]}
        },
    )
    server = interface._InterfaceServer(("127.0.0.1", 0), None)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/", timeout=5) as response:
            page = response.read().decode()
            assert response.status == 200
            assert "Describe the study" in page
            assert "default-src 'self'" in response.headers["Content-Security-Policy"]

        request = Request(
            base + "/api/plan",
            data=json.dumps(
                {
                    "request": "Prepare structural MRI.",
                    "source_root": "/approved/source",
                    "output_root": "/approved/output",
                }
            ).encode(),
            headers={"Content-Type": "application/json", "Origin": base},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            payload = json.load(response)
            assert response.status == 200
            assert payload["result"]["ready"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_chat_runs_auto_from_one_prompt_and_reports_released_files(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp
) -> None:
    import subprocess
    import sys
    import time

    source = nfs_tmp / "bids"
    source.mkdir()
    output = nfs_tmp / "out"
    # Stand-in for `neuroprep-agentic auto`: writes a report and a release index.
    fake_auto = (
        "import json, pathlib, sys\n"
        f"out = pathlib.Path({str(output)!r})\n"
        "q = pathlib.Path(str(out) + '.neuroprep-qualification'); q.mkdir(parents=True)\n"
        "(q / 'autopilot-report.json').write_text(json.dumps({'outcome': 'released',"
        " 'stages': {'policy': {'status': 'passed'}, 'release': {'status': 'passed'}}}))\n"
        "r = out / 'derivatives' / 'release'; r.mkdir(parents=True)\n"
        "(r / 'model-ready-index.jsonl').write_text(json.dumps({'record_type': 'index_metadata'})"
        " + '\\n' + json.dumps({'path': '/x/sub-01_T1w.nii.gz',"
        " 'executor_id': 'smriprep'}) + '\\n')\n"
        "print('auto', sys.argv[1:])\n"
    )
    real_popen = subprocess.Popen
    commands = []

    def popen(command, **kwargs):
        commands.append(command)
        return real_popen([sys.executable, "-c", fake_auto, *command[3:]], **kwargs)

    monkeypatch.setattr(interface.AutoRuns, "launch", staticmethod(popen))
    server = interface._InterfaceServer(("127.0.0.1", 0), None)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/chat", timeout=5) as response:
            assert "NeuroPrep Chat" in response.read().decode()
        text = f"Preprocess T1w. My data is at {source}; save outputs to {output}."
        request = Request(
            base + "/api/auto",
            data=json.dumps({"request": text, "execute": "local"}).encode(),
            headers={"Content-Type": "application/json", "Origin": base},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            run = json.load(response)
        assert commands[0][3:7] == ["auto", text, "--execute", "local"]
        for _ in range(100):
            with urlopen(f"{base}/api/runs/{run['run_id']}", timeout=5) as response:
                run = json.load(response)
            if run["state"] != "running":
                break
            time.sleep(0.1)
        assert run["state"] == "finished" and run["report"]["outcome"] == "released"
        assert run["outputs"] == [{"path": "/x/sub-01_T1w.nii.gz", "executor_id": "smriprep"}]
        assert "auto" in run["log_tail"]
    finally:
        server.shutdown()
        server.server_close()


def test_chat_asks_for_missing_or_unusable_data_instead_of_failing(nfs_tmp) -> None:
    runs = interface.AutoRuns(None)
    with pytest.raises(interface.NeedsInput) as asked:
        runs.start({"request": "Preprocess my MRI please."})
    assert asked.value.field == "source_root" and "attach" in asked.value.question

    with pytest.raises(interface.NeedsInput) as asked:
        runs.start({"request": f"Preprocess T1w; save outputs to {nfs_tmp / 'out'}."})
    assert asked.value.field == "source_root"

    with pytest.raises(interface.NeedsInput) as asked:
        runs.start({"request": "Preprocess T1w. My data is at /forbidden/study."})
    assert asked.value.bad_path == "/forbidden/study" and "not allowed" in asked.value.question

    missing = nfs_tmp / "nowhere"
    with pytest.raises(interface.NeedsInput) as asked:
        runs.start({"request": f"Preprocess T1w. My data is at {missing}."})
    assert asked.value.bad_path == str(missing) and "can't find" in asked.value.question


def test_chat_moves_a_used_output_folder_aside_automatically(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp
) -> None:
    source = nfs_tmp / "bids"
    source.mkdir()
    (nfs_tmp / "busy").mkdir()
    (nfs_tmp / "busy" / "x").write_text("x")
    (nfs_tmp / "out").mkdir()  # empty, but an earlier run left its pilot folder next to it
    (nfs_tmp / "out.neuroprep-qualification").mkdir()
    (nfs_tmp / "out.neuroprep-qualification" / "autopilot-report.json").write_text("{}")
    commands = []

    class Done:
        def poll(self):
            return 0

    monkeypatch.setattr(
        interface.AutoRuns,
        "launch",
        staticmethod(lambda command, **kw: commands.append(command) or Done()),
    )
    runs = interface.AutoRuns(None)
    for name in ("busy", "out"):
        run = runs.start({"request": f"My data is at {source}; save outputs to {nfs_tmp / name}."})
        assert run["run_root"] == str(nfs_tmp / f"{name}-2")
        assert run["moved_from"] == str(nfs_tmp / name)
        assert f"save outputs to {nfs_tmp / name}-2." in commands[-1][4]
        assert run["report"] is None  # never the earlier run's report


def test_uploaded_folder_becomes_the_source(monkeypatch: pytest.MonkeyPatch, nfs_tmp) -> None:
    import io

    monkeypatch.setenv("NEUROPREP_OUTPUT_ROOT", str(nfs_tmp / "runs"))
    upload = "0123456789abcdef"
    for name in ("study/sub-01/anat/sub-01_T1w.nii.gz", "study/dataset_description.json"):
        interface.save_upload(upload, name, io.BytesIO(b"data"), 4)
    for bad in ("../escape", "/abs", "a/../../b"):
        with pytest.raises(ValueError):
            interface.save_upload(upload, bad, io.BytesIO(b"data"), 4)
    commands = []

    class Done:
        def poll(self):
            return 0

    monkeypatch.setattr(
        interface.AutoRuns,
        "launch",
        staticmethod(lambda command, **kw: commands.append(command) or Done()),
    )
    run = interface.AutoRuns(None).start({"request": "Preprocess T1w.", "upload_id": upload})

    source = nfs_tmp / "runs" / "uploads" / upload / "study"
    assert run["source_root"] == str(source) and run["uploaded"] is True
    assert (source / "sub-01/anat/sub-01_T1w.nii.gz").read_bytes() == b"data"
    assert commands[0][commands[0].index("--source-root") + 1] == str(source)
    assert Path(run["run_root"]).name.startswith("study-")


def test_chat_saves_to_the_default_output_root_when_the_prompt_names_none(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp
) -> None:
    source = nfs_tmp / "ds000102"
    source.mkdir()
    monkeypatch.setenv("NEUROPREP_OUTPUT_ROOT", str(nfs_tmp / "runs"))
    commands = []

    class Done:
        def poll(self):
            return 0

    monkeypatch.setattr(
        interface.AutoRuns,
        "launch",
        staticmethod(lambda command, **kw: commands.append(command) or Done()),
    )
    run = interface.AutoRuns(None).start({"request": f"Preprocess T1w. My data is at {source}."})

    output = Path(run["run_root"])
    assert output.parent == nfs_tmp / "runs" and output.name.startswith("ds000102-")
    assert run["default_output"] is True
    assert commands[0][commands[0].index("--output-root") + 1] == str(output)


def test_dashboard_reads_live_task_state_logs_and_metrics(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp
) -> None:
    import sqlite3

    from neuroprep_agentic.state import RunState

    output = nfs_tmp / "run"
    database = Path(f"{output}.neuroprep-qualification") / "pilot" / "state" / "tasks.sqlite"
    database.parent.mkdir(parents=True)
    log = nfs_tmp / "smriprep.log"
    log.write_text("line 1\nline 2\n")
    connection = sqlite3.connect(database)
    RunState._schema(connection)
    spec = json.dumps({"subject_id": "sub-01", "stages": ["anat_preprocess"], "metadata": {}})
    connection.executemany(
        "INSERT INTO tasks(task_id,status,executor_id,resource_class,required,fingerprint,"
        "task_json,attempts,updated_at,log_path) VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            ("t-bids", "complete", "bids-validator", "cpu", 1, "f", "{}", 1, "now", None),
            ("t-smri", "running", "smriprep", "cpu", 1, "f", spec, 1, "now", str(log)),
        ],
    )
    connection.executemany(
        "INSERT INTO attempts(task_id,attempt,worker_id,status,started_at,finished_at)"
        " VALUES (?,?,?,?,?,?)",
        [
            ("t-bids", 1, "w", "complete", "2026-11-19T06:00:00Z", "2026-11-19T06:00:02Z"),
            ("t-smri", 1, "w", "running", "2026-11-19T06:00:03+00:00", None),
        ],
    )
    connection.execute("INSERT INTO dependencies(task_id,depends_on) VALUES ('t-smri','t-bids')")
    connection.commit()
    connection.close()

    class Running:
        def poll(self):
            return None

    runs = interface.AutoRuns(None)
    runs._runs["r1"] = {
        "run_id": "r1", "request": "x", "run_root": str(output),
        "log_path": str(nfs_tmp / "run.chat-run.log"), "process": Running(),
    }
    phases = runs.tasks("r1")["phases"]
    assert [p["phase"] for p in phases] == ["pilot"]  # production has not started
    smriprep = next(t for t in phases[0]["tasks"] if t["executor_id"] == "smriprep")
    assert smriprep["status"] == "running" and smriprep["subject_id"] == "sub-01"
    assert smriprep["started_at"] and smriprep["finished_at"] is None
    assert smriprep["depends_on"] == ["t-bids"]
    assert runs.task_log("r1", "pilot", "t-smri")["log"] == "line 1\nline 2"
    with pytest.raises(KeyError):
        runs.task_log("r1", "pilot", "not-a-task")

    sample = interface.SystemMetrics.sample()
    assert "t" in sample and isinstance(sample["gpus"], list)


def test_metrics_sampler_stops_with_its_server_and_ignores_patched_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CIAI 422639: a sampler left running by another test called nvidia-smi through a
    # later test's subprocess.run fake. It must stop with its server and not use the
    # patched global.
    calls = []
    monkeypatch.setattr(interface.subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(interface.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        interface, "_run_process", lambda *a, **k: SimpleNamespace(stdout="")
    )
    assert interface._gpu_stats() == [] and calls == []

    metrics = interface.SystemMetrics(interval=0.01)
    metrics.start()
    metrics.stop()
    time.sleep(0.05)
    count = len(metrics.samples)
    time.sleep(0.05)
    assert len(metrics.samples) == count
