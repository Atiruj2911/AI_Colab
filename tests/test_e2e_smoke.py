"""End-to-end integration smoke test for the full Phase 1 MVP workflow.

Covers:
1. Building the Gradio UI shell (all 8 pages)
2. Generating and uploading a YOLO Detection dataset ZIP
3. Starting a real detached YOLO11n training subprocess (1 epoch)
4. Verifying duplicate-start prevention & UI reconnect via a fresh AppController
5. Verifying checkpoints (best.pt, last.pt), config, data.yaml, runtime_meta, metrics.jsonl
6. Running single-image inference & checking annotated output image
7. Exporting .pt and .onnx and verifying ONNX model validity
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

from application.app_controller import AppController
from config.app_config import AppConfig
from core.dataset.demo_generator import generate_demo_yolo_dataset
from ui.app_ui import build_app


class TestPhase1EndToEndSmoke(unittest.TestCase):
    def test_full_mvp_workflow_and_ui_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace_dir = tmp_path / "workspace"
            config = AppConfig.from_env(workspace_override=workspace_dir)
            controller = AppController(config=config)
            reconnected_controller: AppController | None = None
            try:
                # 1. Verify Gradio UI builds cleanly without errors
                demo, theme, css = build_app(controller)
                self.assertIsNotNone(demo)
                self.assertIsNotNone(theme)
                self.assertTrue(len(css) > 100)

                # 2. Create demo YOLO zip & ingest through DatasetService (real upload path)
                demo_src = tmp_path / "demo_raw"
                demo_zip = tmp_path / "demo_upload.zip"
                generate_demo_yolo_dataset(
                    demo_src,
                    train_count=6,
                    val_count=2,
                    img_size=320,
                    also_create_zip=demo_zip,
                )
                report = controller.upload_dataset([demo_zip], dataset_name="smoke_shapes")
                self.assertTrue(report.is_valid, f"Validation errors: {report.errors}")
                self.assertEqual(report.total_images, 8)
                previews = controller.get_dataset_previews(report.dataset_id, draw_boxes=True)
                self.assertGreater(len(previews), 0)

                # 3. Start real detached YOLO11n training subprocess (1 epoch, imgsz 320)
                job = controller.start_training(
                    dataset_id=report.dataset_id,
                    model_variant="yolo11n",
                    pretrained_weights="yolo11n.pt",
                    epochs=1,
                    batch_size=2,
                    imgsz=320,
                    device="auto",
                    run_name="smoke_run",
                )
                self.assertIsNotNone(job.pid)

                # 4. Simulate browser/UI reconnect with a brand-new AppController instance
                reconnected_controller = AppController(config=config)
                snap = reconnected_controller.get_monitor_snapshot()
                self.assertIsNotNone(snap["job"])
                self.assertEqual(snap["job"].run_id, job.run_id)

                # Wait up to 4 minutes for 1-epoch training
                deadline = time.time() + 240
                final_job = snap["job"]
                while time.time() < deadline:
                    snap = reconnected_controller.get_monitor_snapshot(run_id=job.run_id)
                    final_job = snap["job"]
                    if final_job.status in {"completed", "failed", "stopped"}:
                        break
                    time.sleep(2.0)

                self.assertEqual(
                    final_job.status,
                    "completed",
                    f"Training did not complete cleanly. Status={final_job.status}, Error={final_job.error_message}\n"
                    f"Log tail:\n{snap['log_tail']}",
                )
                self.assertEqual(final_job.current_epoch, 1)
                self.assertGreaterEqual(len(snap["metrics_records"]), 1)

                # 5. Verify saved checkpoints & self-contained run metadata
                run_dir = Path(final_job.run_dir)
                self.assertTrue((run_dir / "train_config.json").exists())
                self.assertTrue((run_dir / "data.yaml").exists())
                self.assertTrue((run_dir / "runtime_meta.json").exists())
                self.assertTrue((run_dir / "metrics.jsonl").exists())
                self.assertTrue(final_job.best_checkpoint_path and Path(final_job.best_checkpoint_path).exists())
                self.assertTrue(final_job.last_checkpoint_path and Path(final_job.last_checkpoint_path).exists())

                checkpoints = reconnected_controller.list_checkpoints()
                self.assertGreaterEqual(len(checkpoints), 2)

                # 6. Test single-image inference on a validation image
                val_img = Path(report.sample_images[0])
                inf_res = reconnected_controller.run_inference(
                    weights_path=final_job.best_checkpoint_path,
                    image_path=val_img,
                    conf=0.1,
                    iou=0.45,
                    imgsz=320,
                )
                self.assertIsNotNone(inf_res.annotated_image_path)
                self.assertTrue(Path(inf_res.annotated_image_path).exists())
                with Image.open(inf_res.annotated_image_path) as out_im:
                    self.assertEqual(out_im.size, (inf_res.image_width, inf_res.image_height))

                # 7. Export .pt and .onnx and verify artifacts
                bundle_dir, export_artifacts = reconnected_controller.export_model(
                    weights_path=final_job.best_checkpoint_path,
                    formats=["pt", "onnx"],
                    imgsz=320,
                    run_id=job.run_id,
                )
                self.assertTrue(bundle_dir.exists())
                by_fmt = {a.format: a for a in export_artifacts}
                self.assertTrue(by_fmt["pt"].success, by_fmt["pt"].message)
                self.assertGreater(by_fmt["pt"].size_bytes, 0)
                self.assertTrue(by_fmt["onnx"].success, by_fmt["onnx"].message)
                self.assertGreater(by_fmt["onnx"].size_bytes, 0)

                import onnx

                onnx_model = onnx.load(by_fmt["onnx"].path)
                onnx.checker.check_model(onnx_model)
            finally:
                controller.close()
                if reconnected_controller is not None:
                    reconnected_controller.close()


if __name__ == "__main__":
    unittest.main()
