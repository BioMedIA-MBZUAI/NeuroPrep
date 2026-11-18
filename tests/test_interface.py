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
