from __future__ import annotations

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .prompt_run import prepare_prompt_run
from .settings import validated_agentic_project

_LOG = logging.getLogger(__name__)
_MAX_REQUEST_BYTES = 32 * 1024
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class InterfacePlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: str = Field(min_length=1, max_length=4000)
    source_root: str = Field(min_length=1, max_length=4096)
    output_root: str = Field(min_length=1, max_length=4096)
    dataset_id: str | None = Field(default=None, max_length=80)
    openjev_review: bool = False

    @field_validator("request", "source_root", "output_root")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("dataset_id")
    @classmethod
    def strip_dataset_id(cls, value: str | None) -> str | None:
        value = value.strip() if value else None
        return value or None


def compile_interface_plan(
    payload: dict[str, Any], *, agentic_project: str | None = None
) -> dict[str, Any]:
    request = InterfacePlanRequest.model_validate(payload)
    source_root = Path(request.source_root).expanduser()
    output_root = Path(request.output_root).expanduser()
    if not source_root.is_absolute():
        raise ValueError("Source folder must be an absolute path")
    if not source_root.exists():
        raise ValueError(f"Source folder does not exist: {source_root}")
    if not source_root.is_dir():
        raise ValueError(f"Source path is not a folder: {source_root}")
    if not output_root.is_absolute():
        raise ValueError("Output folder must be an absolute path")
    result = prepare_prompt_run(
        request.request,
        source_root=str(source_root),
        output_root=str(output_root),
        dataset_id=request.dataset_id,
        agentic_project=agentic_project,
        openjev_review=request.openjev_review,
    )
    output_root = result["binding"]["output_root"]

    def load(name: str) -> Any:
        with Path(output_root, name).open(encoding="utf-8") as stream:
            return json.load(stream)

    return {
        "result": result,
        "policy": load("policy.json"),
        "backend_plan": load("backend-plan.json"),
        "preflight": load("preflight.json"),
    }


def _public_error(exc: Exception) -> str:
    if not isinstance(exc, ValidationError):
        return str(exc)
    messages: list[str] = []
    for error in exc.errors(include_input=False, include_url=False):
        message = str(error["msg"])
        if message.startswith("Value error, "):
            message = message.removeprefix("Value error, ")
        location = ".".join(str(item) for item in error["loc"])
        messages.append(f"{location}: {message}" if location else message)
    return "; ".join(messages)


class _InterfaceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], agentic_project: str | None):
        super().__init__(address, _InterfaceHandler)
        self.agentic_project = agentic_project


class _InterfaceHandler(BaseHTTPRequestHandler):
    server: _InterfaceServer

    def log_message(self, format: str, *args: object) -> None:
        _LOG.info("%s - %s", self.address_string(), format % args)

    def _headers(self, status: HTTPStatus, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()

    def _json(self, status: HTTPStatus, payload: object) -> None:
        rendered = json.dumps(payload, sort_keys=True, default=str).encode()
        self._headers(status, "application/json; charset=utf-8")
        self.wfile.write(rendered)

    def _same_origin(self) -> bool:
        host = urlsplit("//" + self.headers.get("Host", "")).hostname
        if host not in _LOOPBACK_HOSTS:
            return False
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urlsplit(origin)
        return parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS and (
            parsed.port or 80
        ) == self.server.server_port

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/api/health":
            self._json(HTTPStatus.OK, {"ready": True, "service": "neuroprep-interface"})
            return
        asset = _ASSETS.get(path)
        if asset is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        name, content_type = asset
        body = files("neuroprep_agentic.web").joinpath(name).read_bytes()
        self._headers(HTTPStatus.OK, content_type)
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if urlsplit(self.path).path != "/api/plan":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        if not self._same_origin():
            self._json(HTTPStatus.FORBIDDEN, {"error": "Loopback same-origin request required"})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "JSON body required"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length < 1 or length > _MAX_REQUEST_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Invalid request size"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            result = compile_interface_plan(
                payload, agentic_project=self.server.agentic_project
            )
        except (FileNotFoundError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": _public_error(exc)})
            return
        except Exception:
            _LOG.exception("Unexpected interface planning failure")
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Planning failed"})
            return
        self._json(HTTPStatus.OK, result)


def serve_interface(
    *, host: str = "127.0.0.1", port: int = 8787, agentic_project: str | None = None
) -> None:
    if host not in _LOOPBACK_HOSTS:
        raise ValueError("the NeuroPrep interface may bind only to a loopback host")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    project = str(validated_agentic_project(agentic_project))
    server = _InterfaceServer((host, port), project)
    print(f"NeuroPrep interface: http://{host}:{server.server_port}")
    print("Use an SSH tunnel when the server runs on a cluster.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
