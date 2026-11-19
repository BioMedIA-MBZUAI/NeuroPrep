from __future__ import annotations

from importlib.resources import files
import os
from pathlib import Path
import shutil
from typing import Any

import yaml

from .models import DatasetConfig


class RegistryError(ValueError):
    pass


class PipelineRegistry:
    def __init__(self, payload: dict[str, Any]):
        self.version = int(payload.get("version", 1))
        self.runtime = str(payload.get("runtime", "apptainer"))
        self.pipelines = dict(payload.get("pipelines", {}))

    @classmethod
    def default(cls) -> "PipelineRegistry":
        resource = files("neuroprep").joinpath("resources/pipelines.yaml")
        with resource.open("r", encoding="utf-8") as stream:
            return cls(yaml.safe_load(stream))

    @classmethod
    def from_path(cls, path: Path) -> "PipelineRegistry":
        with path.open("r", encoding="utf-8") as stream:
            return cls(yaml.safe_load(stream))

    def spec(self, name: str) -> dict[str, Any]:
        try:
            return dict(self.pipelines[name])
        except KeyError as exc:
            raise RegistryError(f"Pipeline is not registered: {name}") from exc

    def resources(self, name: str) -> dict[str, Any]:
        return dict(self.spec(name).get("resources", {}))

    def command(
        self,
        name: str,
        dataset: DatasetConfig,
        subject: str | None = None,
    ) -> list[str]:
        spec = self.spec(name)
        overrides = dataset.metadata.get("pipeline_commands", {})
        if isinstance(overrides, dict) and name in overrides:
            raw_command = overrides[name]
            if not isinstance(raw_command, list) or not raw_command:
                raise RegistryError(f"pipeline_commands.{name} must be a non-empty list")
            bindings = {
                "subject": (subject or "").removeprefix("sub-"),
                "dataset_id": dataset.dataset_id,
                "root": str(dataset.bids_root),
                "derivatives": str(dataset.derivatives_root),
                "work": str(dataset.work_root),
            }
            return [str(value).format_map(bindings) for value in raw_command]
        mode = spec.get("mode")
        if mode in {"custom", "native"}:
            return []
        bindings = {
            "subject": (subject or "").removeprefix("sub-"),
            "dataset_id": dataset.dataset_id,
            "root": str(dataset.bids_root),
            "derivatives": str(dataset.derivatives_root),
            "work": str(dataset.work_root),
        }
        if mode == "host":
            executable = spec.get("executable")
            if not executable:
                raise RegistryError(f"Host pipeline {name} requires an executable")
            return [
                str(executable),
                *[str(value).format_map(bindings) for value in spec.get("args", [])],
            ]
        image = spec.get("image")
        image_overrides = dataset.metadata.get("pipeline_images", {})
        if isinstance(image_overrides, dict) and name in image_overrides:
            image = image_overrides[name]
            if not isinstance(image, str) or not image.strip():
                raise RegistryError(f"pipeline_images.{name} must be a non-empty path")
        if not image:
            return []
        args = [str(value).format_map(bindings) for value in spec.get("args", [])]
        runtime = os.environ.get("NEUROPREP_CONTAINER_RUNTIME", self.runtime)
        if shutil.which(runtime) is None and runtime == "apptainer" and shutil.which("singularity"):
            runtime = "singularity"
        resource_dir = files("neuroprep").joinpath("resources")
        action = str(spec.get("container_action", "run"))
        if action not in {"run", "exec"}:
            raise RegistryError(
                f"Pipeline {name} has unsupported container_action {action!r}"
            )
        command = [runtime, action]
        if int(spec.get("resources", {}).get("gpus", 0)) > 0:
            command.append("--nv")
        network = spec.get("network")
        if network is not None:
            network = str(network).strip()
            if network not in {"none", "bridge"}:
                raise RegistryError(
                    f"Pipeline {name} has unsupported container network {network!r}"
                )
            command.extend(("--net", "--network", network))
        for binding in spec.get("binds", []):
            source = resource_dir.joinpath(str(binding["source"]))
            target = str(binding["target"])
            suffix = ":ro" if bool(binding.get("readonly", True)) else ""
            command.extend(("--bind", f"{source}:{target}{suffix}"))
        for binding in spec.get("metadata_binds", []):
            key = str(binding["key"])
            source_value = dataset.metadata.get(key)
            if not source_value:
                continue
            source = Path(str(source_value)).expanduser()
            target = str(binding["target"])
            suffix = ":ro" if bool(binding.get("readonly", True)) else ""
            command.extend(("--bind", f"{source}:{target}{suffix}"))
        environment = spec.get("environment", {})
        if not isinstance(environment, dict):
            raise RegistryError(f"Pipeline {name} environment must be a mapping")
        for key, value in sorted(environment.items()):
            command.extend(
                (
                    "--env",
                    f"{key}={str(value).format_map(bindings)}",
                )
            )
        command.extend([
            "--cleanenv",
            "--containall",
            "--bind",
            f"{dataset.bids_root}:/data:ro",
            "--bind",
            f"{dataset.derivatives_root}:/out",
            "--bind",
            f"{dataset.work_root}:/work",
            "--bind",
            f"{resource_dir}:/neuroprep-resources:ro",
            str(image),
        ])
        executable = spec.get("container_executable")
        if action == "exec":
            if not executable:
                raise RegistryError(
                    f"Container exec pipeline {name} requires container_executable"
                )
            command.append(str(executable))
        command.extend(args)
        return command
