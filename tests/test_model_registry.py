"""Unit tests for ModelRegistry and Yolo11DetectionAdapter validation contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.schemas import TrainRequest
from models.registry import build_default_registry


class TestModelRegistry(unittest.TestCase):
    def test_default_registry_resolves_yolo11_detect(self) -> None:
        registry = build_default_registry()
        adapter = registry.get("yolo11", "detect")
        caps = adapter.get_capabilities()
        self.assertEqual(caps.model_family, "yolo11")
        self.assertEqual(caps.task, "detect")
        self.assertIn("pt", caps.supported_export_formats)
        self.assertIn("onnx", caps.supported_export_formats)

    def test_unknown_model_raises_key_error(self) -> None:
        registry = build_default_registry()
        with self.assertRaises(KeyError):
            registry.get("yolov99", "segment")

    def test_validate_train_request_rejects_invalid_options(self) -> None:
        registry = build_default_registry()
        adapter = registry.get("yolo11", "detect")
        with tempfile.TemporaryDirectory() as tmp:
            yaml_file = Path(tmp) / "data.yaml"
            yaml_file.write_text("train: a\nval: b\nnames: [x]\n", encoding="utf-8")

            bad_req = TrainRequest(
                session_id="s1",
                run_id="r1",
                run_name="r1",
                dataset_id="d1",
                dataset_yaml=str(yaml_file),
                model_variant="yolo99z",  # unsupported variant
                mode="invalid_mode",      # unsupported mode
                epochs=0,                 # invalid
                batch_size=999,           # invalid
                imgsz=123,                # unsupported size
            )
            errors = adapter.validate_train_request(bad_req)
            self.assertGreaterEqual(len(errors), 5)


if __name__ == "__main__":
    unittest.main()
