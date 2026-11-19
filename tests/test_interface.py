from __future__ import annotations

import json
import threading
from pathlib import Path
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

    monkeypatch.setattr(interface.subprocess, "Popen", popen)
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


def test_chat_rejects_a_prompt_without_paths() -> None:
    runs = interface.AutoRuns(None)
    with pytest.raises(ValueError):
        runs.start({"request": "Preprocess my MRI please."})
