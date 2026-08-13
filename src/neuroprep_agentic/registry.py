from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import ClassificationCandidate, SourceRecord
from .plugins.base import DatasetAdapter, DetectionContext, ModalityPlugin
from .plugins.datasets import builtin_dataset_adapters
from .plugins.modalities import builtin_modality_plugins


@dataclass
class PluginRegistry:
    datasets: dict[str, DatasetAdapter] = field(default_factory=dict)
    modalities: dict[str, ModalityPlugin] = field(default_factory=dict)

    def register_dataset(self, adapter: DatasetAdapter) -> None:
        if adapter.adapter_id in self.datasets:
            raise ValueError(f"Duplicate dataset adapter: {adapter.adapter_id}")
        self.datasets[adapter.adapter_id] = adapter

    def register_modality(self, plugin: ModalityPlugin) -> None:
        if plugin.plugin_id in self.modalities:
            raise ValueError(f"Duplicate modality plugin: {plugin.plugin_id}")
        self.modalities[plugin.plugin_id] = plugin

    def dataset_adapter(self, dataset_id: str) -> DatasetAdapter:
        for adapter in self.datasets.values():
            if adapter.recognizes(dataset_id):
                return adapter
        return self.datasets["generic"]

    def modality_plugin(self, plugin_id: str) -> ModalityPlugin:
        try:
            return self.modalities[plugin_id]
        except KeyError as exc:
            raise KeyError(f"Unknown modality plugin: {plugin_id}") from exc

    def detect(self, record: SourceRecord) -> list[ClassificationCandidate]:
        context = DetectionContext(record)
        candidates = [
            candidate
            for plugin in self.modalities.values()
            if (candidate := plugin.detect(context)) is not None
        ]
        return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.plugin_id))

    def inventory(self) -> dict[str, Any]:
        support: dict[str, int] = {}
        modalities = []
        for plugin in sorted(self.modalities.values(), key=lambda item: item.plugin_id):
            key = str(plugin.support_level)
            support[key] = support.get(key, 0) + 1
            modalities.append(
                {
                    "plugin_id": plugin.plugin_id,
                    "modality": plugin.modality,
                    "bids_datatype": plugin.bids_datatype,
                    "bids_suffix": plugin.bids_suffix,
                    "support_level": plugin.support_level,
                }
            )
        return {
            "dataset_adapters": sorted(self.datasets),
            "modality_plugins": modalities,
            "support_counts": support,
        }


def builtin_registry() -> PluginRegistry:
    registry = PluginRegistry()
    for adapter in builtin_dataset_adapters():
        registry.register_dataset(adapter)
    for plugin in builtin_modality_plugins():
        registry.register_modality(plugin)
    return registry
