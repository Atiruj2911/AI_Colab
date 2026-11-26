"""Upload guard tests: unsupported archives and non-YOLO ZIPs are rejected without leaving junk datasets."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from config.app_config import AppConfig
from core.dataset.demo_generator import generate_demo_yolo_dataset
from infrastructure.storage.workspace_manager import WorkspaceManager
from application.services.dataset_service import DatasetService


def _make_service(tmp: Path) -> tuple[DatasetService, WorkspaceManager]:
    config = AppConfig.from_env(workspace_override=tmp / "ws")
    workspace = WorkspaceManager(config.workspace_root)
    return DatasetService(config, workspace), workspace


class TestUploadRejection(unittest.TestCase):
    def test_rar_upload_is_rejected_before_creating_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service, workspace = _make_service(Path(tmp))
            rar = Path(tmp) / "dataset.rar"
            rar.write_bytes(b"Rar!\x1a\x07\x00 not really a rar")
            with self.assertRaises(ValueError) as ctx:
                service.ingest_upload([rar])
            self.assertIn(".zip", str(ctx.exception))
            self.assertEqual(list(workspace.datasets_dir.iterdir()), [])

    def test_zip_without_yaml_or_images_is_rejected_and_cleaned_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service, workspace = _make_service(Path(tmp))
            zip_path = Path(tmp) / "dataset.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("dataset/images/train/", "")
                zf.writestr("dataset/labels/train/", "")
                zf.writestr("dataset/labels/train.cache", "cache")

            with self.assertRaises(ValueError) as ctx:
                service.ingest_upload([zip_path])
            msg = str(ctx.exception)
            self.assertIn("data.yaml", msg)
            self.assertIn("ไฟล์ภาพ", msg)
            # Extracted copy removed, original archive preserved for inspection
            self.assertEqual(list(workspace.datasets_dir.iterdir()), [])
            self.assertTrue(any(workspace.uploads_dir.rglob("dataset.zip")))

    def test_valid_demo_zip_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service, workspace = _make_service(Path(tmp))
            src = Path(tmp) / "src"
            zip_path = Path(tmp) / "good.zip"
            generate_demo_yolo_dataset(src, train_count=2, val_count=1, img_size=128, also_create_zip=zip_path)
            report = service.ingest_upload([zip_path], dataset_name="good")
            self.assertTrue(report.is_valid, report.errors)
            self.assertEqual(report.total_images, 3)
            self.assertEqual(len(list(workspace.datasets_dir.iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
