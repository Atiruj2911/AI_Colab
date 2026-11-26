"""Model registry mapping (model_family, task) -> BaseModelAdapter."""

from __future__ import annotations

from models.base_adapter import BaseModelAdapter, ModelCapabilities


class ModelRegistry:
    """Extensible registry enabling Phase 4 model/task expansion without modifying services."""

    def __init__(self) -> None:
        self._adapters: dict[tuple[str, str], BaseModelAdapter] = {}

    def register(self, adapter: BaseModelAdapter) -> None:
        caps = adapter.get_capabilities()
        key = (caps.model_family.lower(), caps.task.lower())
        self._adapters[key] = adapter

    def get(self, model_family: str, task: str) -> BaseModelAdapter:
        key = (model_family.lower(), task.lower())
        if key not in self._adapters:
            available = ", ".join(f"{f}:{t}" for f, t in sorted(self._adapters))
            raise KeyError(
                f"ไม่รองรับโมเดล '{model_family}' สำหรับงาน '{task}' ในเวอร์ชันนี้ "
                f"(ที่รองรับตอนนี้: {available or 'none'})"
            )
        return self._adapters[key]

    def list_capabilities(self) -> list[ModelCapabilities]:
        return [adapter.get_capabilities() for adapter in self._adapters.values()]


def build_default_registry() -> ModelRegistry:
    """Register Phase 1 adapters (YOLO11 Detection)."""
    from models.yolo.yolo_adapter import Yolo11DetectionAdapter

    registry = ModelRegistry()
    registry.register(Yolo11DetectionAdapter())
    return registry
