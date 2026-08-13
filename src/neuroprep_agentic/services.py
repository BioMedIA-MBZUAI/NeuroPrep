from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from .bids import bids_relative_path
from .identity import IdentityResolver
from .ingestion import IngestionEngine
from .models import DatasetSpec, SourceRecord
from .planning import PipelinePlanner
from .registry import PluginRegistry, builtin_registry
from .settings import validated_cluster_path


class ManifestDocument(BaseModel):
    dataset: DatasetSpec
    records: list[SourceRecord] = Field(max_length=100_000)
    explicit_identity_crosswalk: dict[str, str] = Field(default_factory=dict)


def load_manifest(path: str) -> ManifestDocument:
    candidate = validated_cluster_path(path, must_exist=True)
    if candidate.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("Manifest exceeds the 512 MiB safety limit")
    with candidate.open("r", encoding="utf-8") as stream:
        return ManifestDocument.model_validate(json.load(stream))


def build_graph_and_plan(
    manifest: ManifestDocument,
    identity_key: str,
    registry: PluginRegistry | None = None,
):
    selected_registry = registry or builtin_registry()
    identity = IdentityResolver(identity_key, manifest.explicit_identity_crosswalk)
    graph = IngestionEngine(selected_registry, identity).build_graph(
        manifest.dataset, manifest.records
    )
    return graph, PipelinePlanner(selected_registry).plan(graph)


def build_ingest_plan(
    manifest: ManifestDocument,
    identity_key: str,
    registry: PluginRegistry | None = None,
) -> dict[str, Any]:
    graph, plan = build_graph_and_plan(manifest, identity_key, registry)
    public_plan = plan.model_dump(mode="json")
    for issue in public_plan["issues"]:
        # Agents receive the stable acquisition ID, never a source path that may
        # embed a patient identifier. The internal graph retains the path.
        issue["path"] = None
    mappings = [
        {
            "acquisition_id": acquisition.acquisition_id,
            "subject_id": acquisition.subject_id,
            "session_id": acquisition.session_id,
            "plugin_id": acquisition.selected_plugin_id,
            "status": acquisition.status,
            "bids_path": (
                str(bids_relative_path(acquisition)) if acquisition.bids is not None else None
            ),
        }
        for acquisition in graph.acquisitions
    ]
    return {
        "schema_version": graph.schema_version,
        "dataset_id": manifest.dataset.dataset_id,
        "dataset_adapter": graph.provenance["dataset_adapter"],
        "subjects": len(graph.subjects),
        "acquisitions": len(graph.acquisitions),
        "quarantined": len(graph.quarantined),
        "unsupported": len(graph.unsupported),
        "mappings": mappings,
        "plan": public_plan,
    }
