"""Unit tests for safe ZIP extraction and YOLO Detection dataset validation."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from core.dataset.demo_generator import generate_demo_yolo_dataset
from core.dataset.validator import validate_yolo_dataset
from infrastructure.storage.archive_utils import ArchiveSecurityError, safe_extract_zip


class TestDatasetValidation(unittest.TestCase):
    def test_valid_demo_dataset_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "demo"
            generate_demo_yolo_dataset(root, train_count=4, val_count=2, img_size=160)
            report = validate_yolo_dataset("ds_1", "demo", root)
            self.assertTrue(report.is_valid, f"Unexpected errors: {report.errors}")
            self.assertEqual(report.total_images, 6)
            self.assertEqual(report.total_labels, 6)
            self.assertEqual(report.class_names, {0: "box", 1: "circle"})
            self.assertIsNotNone(report.resolved_yaml_path)
            self.assertTrue(Path(report.resolved_yaml_path).exists())

    def test_missing_data_yaml_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "empty_ds"
            root.mkdir()
            report = validate_yolo_dataset("ds_missing", "empty", root)
            self.assertFalse(report.is_valid)
            codes = {e.code for e in report.errors}
            self.assertIn("MISSING_DATA_YAML", codes)

    def test_invalid_yaml_syntax_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bad_yaml"
            root.mkdir()
            (root / "data.yaml").write_text("names: [a, b\ntrain: : bad", encoding="utf-8")
            report = validate_yolo_dataset("ds_yaml", "bad_yaml", root)
            self.assertFalse(report.is_valid)
            codes = {e.code for e in report.errors}
            self.assertIn("INVALID_YAML", codes)

    def test_corrupt_image_and_malformed_label_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "corrupt_ds"
            generate_demo_yolo_dataset(root, train_count=3, val_count=2, img_size=128)
            # Corrupt one image
            (root / "images" / "train" / "demo_train_000.jpg").write_bytes(b"not-a-valid-jpeg")
            # Malformed label (out-of-range class id & bad coordinates)
            (root / "labels" / "train" / "demo_train_001.txt").write_text("99 1.5 0.5 0.2 0.2\n", encoding="utf-8")
            # Empty label (warning only)
            (root / "labels" / "val" / "demo_val_000.txt").write_text("", encoding="utf-8")

            report = validate_yolo_dataset("ds_bad", "corrupt", root)
            self.assertFalse(report.is_valid)
            err_codes = {e.code for e in report.errors}
            warn_codes = {w.code for w in report.warnings}
            self.assertIn("UNREADABLE_IMAGES_TRAIN", err_codes)
            self.assertIn("MALFORMED_LABELS_TRAIN", err_codes)
            self.assertIn("EMPTY_LABELS_VAL", warn_codes)

    def test_zip_slip_path_traversal_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "evil.zip"
            dest = Path(tmp) / "out"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("../escape.txt", "malicious")
            with self.assertRaises(ArchiveSecurityError):
                safe_extract_zip(zip_path, dest)

    def test_safe_zip_extract_unwraps_single_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src_ds = Path(tmp) / "src"
            generate_demo_yolo_dataset(src_ds, train_count=2, val_count=1, img_size=128)
            zip_path = Path(tmp) / "wrapped.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for p in src_ds.rglob("*"):
                    if p.is_file():
                        zf.write(p, arcname=f"my_wrapped_ds/{p.relative_to(src_ds)}")

            extracted = safe_extract_zip(zip_path, Path(tmp) / "dest")
            self.assertEqual(extracted.name, "my_wrapped_ds")
            self.assertTrue((extracted / "data.yaml").exists())


if __name__ == "__main__":
    unittest.main()
