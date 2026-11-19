from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from . import paths
from .autopilot import qualification_root
from .prompt_run import extract_prompt_paths, prepare_prompt_run, resolve_prompt_paths
from .settings import validated_agentic_project, validated_cluster_path

_LOG = logging.getLogger(__name__)
_MAX_REQUEST_BYTES = 32 * 1024
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/chat": ("chat.html", "text/html; charset=utf-8"),
    "/chat.css": ("chat.css", "text/css; charset=utf-8"),
    "/chat.js": ("chat.js", "text/javascript; charset=utf-8"),
}
_EXECUTE_MODES = ("local", "slurm", "plan-only")


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


class InterfaceAutoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: str = Field(min_length=1, max_length=4000)
    execute: str = "local"
    upload_id: str | None = None

    @field_validator("execute")
    @classmethod
    def known_mode(cls, value: str) -> str:
        if value not in _EXECUTE_MODES:
            raise ValueError(f"execute must be one of {', '.join(_EXECUTE_MODES)}")
        return value


class NeedsInput(Exception):
    """A question for the person in the chat instead of an error.

    ``bad_path`` is the text in their request that the answer should replace.
    """

    def __init__(self, field: str, question: str, bad_path: str | None = None) -> None:
        super().__init__(question)
        self.field = field
        self.question = question
        self.bad_path = bad_path

    def payload(self) -> dict[str, Any]:
        return {"needs": self.field, "question": self.question, "bad_path": self.bad_path}


_ASK_SOURCE = (
    "Where is the data? Paste the folder path (for example /data/my-study), "
    "or attach the folder or files."
)


def _roots() -> str:
    return ", ".join(str(root) for root in paths.allowed_roots())


def _resolve_chat_paths(text: str, uploaded: Path | None) -> tuple[Any, bool]:
    """Paths from the request, checked one by one so a problem becomes a question.

    A request without an output folder gets a new one under the output root.
    """

    try:
        typed_source, typed_output = extract_prompt_paths(
            text, source_root=str(uploaded) if uploaded else None
        )
    except ValueError as error:  # the request names a folder and files were attached
        raise NeedsInput(
            "source_root",
            "You attached files and also named a data folder. Remove the folder from your "
            "message to use the attached files, or remove the attachment.",
        ) from error
    if not typed_source:
        raise NeedsInput("source_root", _ASK_SOURCE)
    typed = None if uploaded else typed_source
    source = Path(typed_source).expanduser()
    _check_source(source, typed)
    source = source.resolve()
    default_output = not typed_output
    if default_output:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = paths.output_root() / f"{source.name}-{stamp}"
    else:
        output = Path(str(typed_output)).expanduser()
    _check_output(output, None if default_output else typed_output, source)
    binding = resolve_prompt_paths(text, source_root=str(source), output_root=str(output))
    return binding, default_output


def _check_source(source: Path, typed: str | None) -> None:
    if not source.is_absolute():
        raise NeedsInput(
            "source_root", f"Please give the full path of {source}, starting with /.", typed
        )
    try:
        validated_cluster_path(str(source))
    except ValueError:
        raise NeedsInput(
            "source_root",
            f"I'm not allowed to read {source}: data must be under {_roots()}. "
            "Give a folder there, or attach the files.",
            typed,
        ) from None
    if not source.exists():
        raise NeedsInput(
            "source_root", f"I can't find {source}. Check the path, or attach the files.", typed
        )
    if not source.is_dir():
        raise NeedsInput(
            "source_root",
            f"{source} is a file. Give the folder that contains the data, or attach it.",
            typed,
        )


def _check_output(output: Path, typed: str | None, source: Path) -> None:
    def ask(problem: str) -> NeedsInput:
        return NeedsInput(
            "output_root", f"{problem} Which new folder should I save the results to?", typed
        )

    if not output.is_absolute():
        raise ask(f"{output} is not a full path (it must start with /).")
    try:
        validated_cluster_path(str(output))
    except ValueError:
        raise ask(f"I can't save to {output}: results must go under {_roots()}.") from None
    resolved = output.resolve()
    if resolved in paths.allowed_roots():
        raise ask(f"{output} is a whole storage area, not a results folder.")
    if resolved == source or source in resolved.parents or resolved in source.parents:
        raise ask(f"{output} overlaps the data folder {source}.")
    if resolved.exists() and any(resolved.iterdir()):
        raise ask(f"{output} already contains files.")


_UPLOAD_ID = re.compile(r"^[a-f0-9]{16,32}$")
_MAX_UPLOAD_BYTES = int(os.environ.get("NEUROPREP_MAX_UPLOAD_MB", "20480")) * 1024 * 1024


def upload_dir(upload_id: str) -> Path:
    if not _UPLOAD_ID.match(upload_id):
        raise ValueError("invalid upload id")
    return paths.output_root() / "uploads" / upload_id


def save_upload(upload_id: str, relative: str, stream: Any, length: int) -> Path:
    """Write one uploaded file under the upload folder, keeping its relative folders."""

    parts = PurePosixPath(relative).parts
    if not parts or relative.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise ValueError("invalid file path")
    if length > _MAX_UPLOAD_BYTES:
        raise ValueError("file is larger than NEUROPREP_MAX_UPLOAD_MB")
    root = upload_dir(upload_id)
    target = root.joinpath(*parts)
    validated_cluster_path(str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    remaining = length
    with partial.open("wb") as handle:
        while remaining:
            chunk = stream.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("upload ended early")
            handle.write(chunk)
            remaining -= len(chunk)
    partial.replace(target)
    return target


def upload_source(upload_id: str) -> Path:
    """The uploaded data folder: the single top folder of a folder upload, else the root."""

    root = upload_dir(upload_id)
    entries = list(root.iterdir()) if root.is_dir() else []
    if not entries:
        raise NeedsInput("source_root", "The attached files did not arrive. Attach them again.")
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return root


class AutoRuns:
    """`neuroprep-agentic auto` runs started from the chat page, one subprocess each.

    shortcut: runs are tracked in memory, so a server restart forgets them (their
    reports and outputs stay on disk); persist the table if the server must restart.
    """

    def __init__(self, agentic_project: str | None) -> None:
        self.agentic_project = agentic_project
        self._runs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = InterfaceAutoRequest.model_validate(payload)
        uploaded = upload_source(request.upload_id) if request.upload_id else None
        binding, default_output = _resolve_chat_paths(request.request, uploaded)
        source = Path(binding.source_root)
        output = Path(binding.output_root)
        log_path = Path(f"{output}.chat-run.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-m", "neuroprep_agentic.main", "auto", request.request,
            "--execute", request.execute,
        ]
        if uploaded is not None:
            command += ["--source-root", str(uploaded)]
        if default_output:
            command += ["--output-root", str(output)]
        if self.agentic_project:
            command += ["--agentic-project", self.agentic_project]
        with log_path.open("wb") as log:
            process = subprocess.Popen(  # noqa: S603 - argv list, no shell
                command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ)
            )
        run_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._runs[run_id] = {
                "run_id": run_id,
                "request": request.request,
                "execute": request.execute,
                "source_root": str(source),
                "run_root": str(output),
                "default_output": default_output,
                "uploaded": uploaded is not None,
                "log_path": str(log_path),
                "process": process,
            }
        return self.status(run_id)

    def status(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        exit_code = run["process"].poll()
        report_path = qualification_root(Path(run["run_root"])) / "autopilot-report.json"
        report = _read_json(report_path)
        state = "running" if exit_code is None else ("finished" if exit_code == 0 else "failed")
        index = Path(run["run_root"]) / "derivatives" / "release" / "model-ready-index.jsonl"
        return {
            **{key: value for key, value in run.items() if key != "process"},
            "state": state,
            "exit_code": exit_code,
            "report": report,
            "log_tail": _tail(Path(run["log_path"])),
            "outputs": _released_outputs(index),
        }

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(self._runs)
        return [self.status(run_id) for run_id in ids]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _released_outputs(index: Path, limit: int = 200) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with index.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("record_type") == "index_metadata":
                    continue
                rows.append(row)
                if len(rows) >= limit:
                    break
    except (OSError, ValueError):
        return []
    return rows


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
        self.runs = AutoRuns(agentic_project)


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
        if path == "/api/runs":
            self._json(HTTPStatus.OK, {"runs": self.server.runs.list()})
            return
        if path.startswith("/api/runs/"):
            try:
                self._json(HTTPStatus.OK, self.server.runs.status(path.removeprefix("/api/runs/")))
            except KeyError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown run"})
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
        url = urlsplit(self.path)
        route = url.path
        if route not in {"/api/plan", "/api/auto", "/api/upload"}:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        if not self._same_origin():
            self._json(HTTPStatus.FORBIDDEN, {"error": "Loopback same-origin request required"})
            return
        if route == "/api/upload":
            self._upload(parse_qs(url.query))
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
            if route == "/api/auto":
                try:
                    result = self.server.runs.start(payload)
                except NeedsInput as question:
                    result = question.payload()
            else:
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


    def _upload(self, query: dict[str, list[str]]) -> None:
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if length < 0:
                raise ValueError("Content-Length required")
            upload_id = query.get("id", [""])[0]
            relative = query.get("path", [""])[0]
            save_upload(upload_id, relative, self.rfile, length)
        except ValueError as exc:
            self.close_connection = True
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._json(HTTPStatus.OK, {"saved": relative})


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
