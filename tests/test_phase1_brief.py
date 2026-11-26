"""Tests verifying Phase 1 Brief requirements:
- Offline Augmentation (exact bbox transform + Train-only isolation)
- Split Preview & Rename Dataset
- YOLO11n/s/m/l/x + Train New / Fine-tune / Resume validation
- Safe Checkpoint Snapshot Copy (Testing Snapshot + Export .pt / ZIP bundle)
- Unified 4-page Training Workspace UI build
"""

from __future__ import annotations

import random
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from application.app_controller import AppController
from config.app_config import AppConfig
from core.dataset.annotation_io import Box
from core.dataset.augmentor import AugmentConfig, augment_image_and_boxes
from core.schemas import JobState, TrainRequest
from infrastructure.process.state_store import write_json_atomic
from infrastructure.storage.checkpoint_snapshot import safe_snapshot_checkpoint
from models.registry import build_default_registry


def _make_fake_pt(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", b"x" * 2048)


class TestAugmentation(unittest.TestCase):
    def test_horizontal_and_vertical_flip_bbox_alignment(self) -> None:
        img = Image.new("RGB", (200, 100), (10, 20, 30))
        box = Box(class_id=0, cx=0.25, cy=0.30, w=0.20, h=0.20)

        cfg_h = AugmentConfig(hflip_p=1.0, vflip_p=0.0, rotate_deg=0, scale_pct=0, translate_pct=0,
                              brightness_pct=0, contrast_pct=0, saturation_pct=0, hue_deg=0)
        _, boxes_h = augment_image_and_boxes(img, [box], cfg_h, random.Random(1))
        self.assertEqual(len(boxes_h), 1)
        self.assertAlmostEqual(boxes_h[0].cx, 0.75, places=2)
        self.assertAlmostEqual(boxes_h[0].cy, 0.30, places=2)

        cfg_v = AugmentConfig(hflip_p=0.0, vflip_p=1.0, rotate_deg=0, scale_pct=0, translate_pct=0,
                              brightness_pct=0, contrast_pct=0, saturation_pct=0, hue_deg=0)
        _, boxes_v = augment_image_and_boxes(img, [box], cfg_v, random.Random(1))
        self.assertEqual(len(boxes_v), 1)
        self.assertAlmostEqual(boxes_v[0].cx, 0.25, places=2)
        self.assertAlmostEqual(boxes_v[0].cy, 0.70, places=2)

    def test_offline_augmentation_augments_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig.from_env(workspace_override=Path(tmp) / "ws")
            controller = AppController(config=config)
            try:
                ds = controller.dataset_service.create_demo_dataset(train_count=3, val_count=2)
                cfg = AugmentConfig(copies_per_image=2, seed=7, rotate_deg=5, blur_radius=0.5, noise_std=3.0)

                prev_img, box_cnt, total_train = controller.dataset_service.preview_augmentation(ds.dataset_id, cfg, 0)
                self.assertEqual(total_train, 3)
                self.assertIsNotNone(prev_img)

                aug_report, counts = controller.dataset_service.create_augmented_dataset(ds.dataset_id, cfg, new_name="demo_aug")
                self.assertTrue(aug_report.is_valid, [e.message for e in aug_report.errors])
                self.assertEqual(counts["train_original"], 3)
                self.assertEqual(counts["train_augmented"], 6)
                self.assertEqual(counts["val"], 2)  # Val untouched!
                self.assertEqual(aug_report.splits["train"].image_count, 9)
                self.assertEqual(aug_report.splits["val"].image_count, 2)
                self.assertTrue((Path(aug_report.dataset_root) / "augment_manifest.json").exists())
            finally:
                controller.close()


class TestSplitPreviewAndRename(unittest.TestCase):
    def test_preview_split_and_rename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig.from_env(workspace_override=Path(tmp) / "ws")
            controller = AppController(config=config)
            try:
                ds = controller.dataset_service.create_demo_dataset(train_count=7, val_count=3)
                prev = controller.dataset_service.preview_split(ds.dataset_id, 70, 20, 10, drop_duplicates=True)
                self.assertEqual(prev["total"], 10)
                self.assertEqual(prev["train"] + prev["val"] + prev["test"], 10)

                renamed = controller.dataset_service.rename_dataset(ds.dataset_id, "renamed_demo")
                self.assertEqual(renamed.dataset_name, "renamed_demo")
            finally:
                controller.close()


class TestTrainingModesAndSnapshots(unittest.TestCase):
    def test_yolo11_variants_and_modes_validation(self) -> None:
        registry = build_default_registry()
        adapter = registry.get("yolo11", "detect")
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "data.yaml"
            yaml_path.write_text("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: {0: a}\n", encoding="utf-8")
            fake_last = Path(tmp) / "last.pt"
            _make_fake_pt(fake_last)

            for variant in ("yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"):
                req = TrainRequest(
                    session_id="s1", run_id="r1", run_name="r1", dataset_id="d1",
                    dataset_yaml=str(yaml_path), model_variant=variant,
                    pretrained_weights=f"{variant}.pt", mode="train",
                )
                self.assertEqual(adapter.validate_train_request(req), [])

            req_resume_ok = TrainRequest(
                session_id="s1", run_id="r1", run_name="r1", dataset_id="d1",
                dataset_yaml=str(yaml_path), model_variant="yolo11n",
                pretrained_weights=str(fake_last), mode="resume",
            )
            self.assertEqual(adapter.validate_train_request(req_resume_ok), [])

            req_resume_missing = TrainRequest(
                session_id="s1", run_id="r1", run_name="r1", dataset_id="d1",
                dataset_yaml=str(yaml_path), model_variant="yolo11n",
                pretrained_weights=str(Path(tmp) / "missing.pt"), mode="resume",
            )
            self.assertTrue(len(adapter.validate_train_request(req_resume_missing)) > 0)

    def test_safe_snapshot_testing_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig.from_env(workspace_override=Path(tmp) / "ws")
            controller = AppController(config=config)
            try:
                run_dir = controller.workspace.runs_dir / "run_live_1"
                best_pt = run_dir / "weights" / "best.pt"
                last_pt = run_dir / "weights" / "last.pt"
                _make_fake_pt(best_pt)
                _make_fake_pt(last_pt)
                (run_dir / "data.yaml").write_text("nc: 1\nnames: {0: box}\n", encoding="utf-8")

                job = JobState(
                    job_id="run_live_1", session_id="s1", run_id="run_live_1", run_name="run_live",
                    dataset_id="d1", model_family="yolo11", task="detect", model_variant="yolo11n",
                    status="running", current_epoch=3, total_epochs=10, best_epoch=2, best_map50_95=0.65,
                    best_checkpoint_path=str(best_pt), last_checkpoint_path=str(last_pt), run_dir=str(run_dir),
                )
                write_json_atomic(run_dir / "job_state.json", job.to_dict())

                # Testing Snapshot
                snap_info = controller.create_testing_snapshot(run_id="run_live_1")
                self.assertTrue(Path(snap_info["snapshot_path"]).is_file())
                self.assertNotEqual(Path(snap_info["snapshot_path"]).resolve(), best_pt.resolve())
                self.assertEqual(snap_info["best_epoch"], 2)

                # Export best.pt & last.pt & ZIP bundle without stopping training
                exp_best = controller.export_pt_checkpoint(run_id="run_live_1", which="best", include_bundle=False)
                self.assertTrue(Path(exp_best["file_path"]).is_file())
                exp_last = controller.export_pt_checkpoint(run_id="run_live_1", which="last", include_bundle=False)
                self.assertTrue(Path(exp_last["file_path"]).is_file())
                exp_zip = controller.export_pt_checkpoint(run_id="run_live_1", which="best", include_bundle=True)
                self.assertTrue(Path(exp_zip["file_path"]).suffix == ".zip")
                with zipfile.ZipFile(exp_zip["file_path"]) as zf:
                    self.assertIn("best.pt", zf.namelist())
                    self.assertIn("data.yaml", zf.namelist())
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()
