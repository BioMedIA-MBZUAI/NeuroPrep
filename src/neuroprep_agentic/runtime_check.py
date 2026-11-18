"""Deployment-host readiness checks with explicit pass/fail evidence."""

from __future__ import annotations

import os
import shutil
import urllib.error
import urllib.request
from typing import Any

from . import paths


def _endpoint_check(url: str, timeout: float = 3.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return {"ready": True, "status": response.status, "url": url}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"ready": False, "url": url, "error": str(exc)}


def _llm_check() -> dict[str, Any]:
    """Readiness of the configured manager model, whatever the provider."""

    from .settings import AgenticSettings

    try:
        settings = AgenticSettings.from_env()
        endpoint = settings.endpoint(settings.manager_model)
    except ValueError as exc:
        return {"ready": False, "error": str(exc)}
    base = {"provider": endpoint.provider, "model": endpoint.model}
    if endpoint.provider in {"openai", "openrouter", "anthropic"}:
        # Hosted: never spend tokens or leak the key in a readiness probe; key presence only.
        return {**base, "ready": bool(endpoint.api_key), "credentials": "present"}
    probe = "/api/tags" if endpoint.provider == "ollama" else "/models"
    return {**base, **_endpoint_check(endpoint.base_url + probe)}


def runtime_check() -> dict[str, Any]:
    fsl_root = paths.fsl_mrs_root()
    ants_image = paths.ants_image()
    checks = {
        "llm": _llm_check(),
        "openjev": _endpoint_check(
            os.getenv(
                "NEUROPREP_OPENJEV_ENDPOINT",
                "http://127.0.0.1:3000/v1/systemone",
            ).rsplit("/v1/", 1)[0]
            + "/v1/version"
        ),
        "fsl_mrs": {
            "ready": all(
                (fsl_root / "bin" / name).is_file()
                for name in ("spec2nii", "fsl_mrs_preproc", "fsl_mrs")
            ),
            "root": str(fsl_root),
            "executables": {
                name: (fsl_root / "bin" / name).is_file()
                for name in ("spec2nii", "fsl_mrs_preproc", "fsl_mrs")
            },
        },
        "ants": {"ready": ants_image.is_file(), "image": str(ants_image)},
        "container_runtime": {
            "ready": bool(shutil.which("apptainer") or shutil.which("singularity")),
            "executable": shutil.which("apptainer") or shutil.which("singularity"),
        },
    }
    return {"ready": all(bool(item.get("ready")) for item in checks.values()), "checks": checks}
