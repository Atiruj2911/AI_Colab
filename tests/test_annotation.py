"""Phase 2 tests: label editor safety, classes, diagnostics, split, import/export, accept proposals."""

from __future__ import annotations

import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from application.services.annotation_service import AnnotationService
from application.services.dataset_service import DatasetService
from config.app_config import AppConfig
from core.dataset.annotation_io import boxes_to_pixel_rows, parse_label_file, pixel_rows_to_boxes
from core.dataset.demo_generator import generate_demo_yolo_dataset
from infrastructure.process.state_store import write_json_atomic
from infrastructure.storage.workspace_manager import WorkspaceManager
from models.registry import build_default_registry


def _setup(tmp: Path):
    config = AppConfig.from_env(workspace_override=tmp / "ws")
    workspace = WorkspaceManager(config.workspace_root)
    datasets = DatasetService(config, workspace)
    annotations = AnnotationService(workspace, datasets, build_default_registry())
    src = tmp / "src"
    zip_path = tmp / "demo.zip"
    generate_demo_yolo_dataset(src, train_count=4, val_count=2, img_size=128, also_create_zip=zip_path)
    report = datasets.ingest_upload([zip_path], dataset_name="demo")
    return datasets, annotations, workspace, report


class TestCoordinateRoundtrip(unittest.TestCase):
    def test_pixel_normalized_roundtrip(self) -> None:
        names = {"box": 0, "circle": 1}
        rows = [["box", 10, 20, 110, 70], ["circle", 0, 0, 50, 40]]
        boxes = pixel_rows_to_boxes(rows, 200, 100, names)
        self.assertEqual(len(boxes), 2)
        back = boxes_to_pixel_rows(boxes, {0: "box", 1: "circle"}, 200, 100)
        self.assertEqual([r[1:] for r in back], [[10, 20, 110, 70], [0, 0, 50, 40]])
        self.assertEqual([r[0] for r in back], ["box", "circle"])

    def test_reversed_box_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pixel_rows_to_boxes([["box", 100, 20, 10, 70]], 200, 100, {"box": 0})

    def test_unknown_class_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pixel_rows_to_boxes([["triangle", 1, 1, 50, 50]], 200, 100, {"box": 0})


class TestLabelEditing(unittest.TestCase):
    def test_save_guard_and_single_undo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            datasets, ann, _, report = _setup(Path(tmp))
            key = ann.list_images(report.dataset_id, "train")[0]["key"]
            editor = ann.load_editor(report.dataset_id, "train", key)
            original_rows = editor["rows"]
            self.assertGreater(len(original_rows), 0, "demo images should have boxes")

            new_rows = original_rows + [["circle", 5, 5, 30, 30]]
            with self.assertRaises(ValueError):  # existing label without confirmation
                ann.save_boxes(report.dataset_id, "train", key, new_rows, allow_overwrite=False)

            ann.save_boxes(report.dataset_id, "train", key, new_rows, allow_overwrite=True)
            label = ann.load_editor(report.dataset_id, "train", key)
            self.assertEqual(len(label["rows"]), len(original_rows) + 1)

            ann.undo_last_save(report.dataset_id, "train", key)
            restored = ann.load_editor(report.dataset_id, "train", key)
            self.assertEqual(len(restored["rows"]), len(original_rows))
            with self.assertRaises(ValueError):  # only one level of undo
                ann.undo_last_save(report.dataset_id, "train", key)

    def test_unlabeled_image_save_and_undo_removes_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            datasets, ann, _, report = _setup(Path(tmp))
            entry = ann.list_images(report.dataset_id, "val")[0]
            key = entry["key"]
            label_path = Path(report.splits["val"].labels_dir) / f"{key}.txt"
            label_path.unlink()

            ann.save_boxes(report.dataset_id, "val", key, [["box", 1, 1, 40, 40]], allow_overwrite=False)
            self.assertEqual(len(parse_label_file(label_path)[0]), 1)
            ann.undo_last_save(report.dataset_id, "val", key)
            self.assertFalse(label_path.exists())

    def test_rename_and_delete_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            datasets, ann, _, report = _setup(Path(tmp))
            key = ann.list_images(report.dataset_id, "train")[0]["key"]
            new_key = ann.rename_image(report.dataset_id, "train", key, "renamed_01")
            self.assertEqual(new_key, "renamed_01")
            images_dir = Path(report.splits["train"].images_dir)
            self.assertTrue(any(p.stem == "renamed_01" for p in images_dir.iterdir()))

            with self.assertRaises(ValueError):
                ann.rename_image(report.dataset_id, "train", "renamed_01", "../evil")

            with self.assertRaises(ValueError):  # needs explicit confirm
                ann.delete_image(report.dataset_id, "train", "renamed_01", confirm=False)
            ann.delete_image(report.dataset_id, "train", "renamed_01", confirm=True)
            self.assertFalse(any(p.stem == "renamed_01" for p in images_dir.iterdir()))


class TestClasses(unittest.TestCase):
    def test_add_rename_remove_last(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            datasets, ann, _, report = _setup(Path(tmp))
            ds_id = report.dataset_id

            ann.add_class(ds_id, "square")
            names = {c["id"]: c["name"] for c in ann.list_classes(ds_id)}
            self.assertEqual(names[2], "square")

            self.assertEqual(ann.rename_classes(ds_id, {0: "box_new"}), 1)
            names = {c["id"]: c["name"] for c in ann.list_classes(ds_id)}
            self.assertEqual(names[0], "box_new")

            with self.assertRaises(ValueError):  # not the highest id
                ann.remove_last_class(ds_id, 0)
            ann.remove_last_class(ds_id, 2)  # unused and highest
            self.assertEqual(len(ann.list_classes(ds_id)), 2)

            with self.assertRaises(ValueError):  # class 1 is used by demo boxes
                ann.remove_last_class(ds_id, 1)


class TestDiagnostics(unittest.TestCase):
    def test_duplicates_and_orphans_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            datasets, ann, _, report = _setup(Path(tmp))
            images_dir = Path(report.splits["train"].images_dir)
            labels_dir = Path(report.splits["train"].labels_dir)
            first = sorted(images_dir.iterdir())[0]
            shutil.copy2(first, images_dir / ("copy_of_" + first.name))
            (labels_dir / "ghost.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            d = ann.diagnostics(report.dataset_id)
            self.assertGreaterEqual(len(d["duplicates"]), 1)
            self.assertTrue(any("ghost" in str(x) for x in d["orphan_labels"]))


class TestSplitAndIO(unittest.TestCase):
    def test_split_creates_new_dataset_with_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            datasets, ann, workspace, report = _setup(Path(tmp))
            new_report, counts = ann.create_split_dataset(
                dataset_id=report.dataset_id,
                train_pct=50,
                val_pct=50,
                test_pct=0,
                seed=7,
                drop_duplicates=True,
                new_name="split_test",
            )
            self.assertEqual(counts["train"] + counts["val"], report.total_images)
            self.assertEqual(counts["test"], 0)
            self.assertTrue(new_report.is_valid, [e.message for e in new_report.errors])
            self.assertNotEqual(new_report.dataset_id, report.dataset_id)
            self.assertTrue((Path(new_report.dataset_root) / "split_manifest.json").exists())

    def test_split_rejects_bad_percentages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            datasets, ann, _, report = _setup(Path(tmp))
            with self.assertRaises(ValueError):
                ann.create_split_dataset(report.dataset_id, 60, 30, 0, 1, True, None)

    def test_export_then_import_restores_missing_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            datasets, ann, _, report = _setup(Path(tmp))
            zip_path = ann.export_labels(report.dataset_id)
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
            self.assertIn("data.yaml", names)
            self.assertIn("classes.txt", names)

            victim = Path(report.splits["train"].labels_dir) / (
                ann.list_images(report.dataset_id, "train")[0]["key"] + ".txt"
            )
            victim.unlink()
            result = ann.import_labels(report.dataset_id, zip_path, allow_overwrite=False)
            self.assertEqual(result["imported"], 1)
            self.assertEqual(result["unchanged"], report.total_images - 1)  # every other label is identical
            self.assertTrue(victim.exists())


class TestAcceptProposals(unittest.TestCase):
    def test_accept_writes_only_unlabeled_when_not_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            datasets, ann, _, report = _setup(Path(tmp))
            ds_id = report.dataset_id
            train_list = ann.list_images(ds_id, "train")
            empty_key = train_list[0]["key"]
            labeled_key = train_list[1]["key"]
            label_empty = Path(report.splits["train"].labels_dir) / f"{empty_key}.txt"
            label_empty.write_text("", encoding="utf-8")  # unlabeled (background) now

            box = {"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "conf": 0.9}
            proposal = {
                "dataset_id": ds_id,
                "items": [
                    {"split": "train", "key": empty_key, "boxes": [box]},
                    {"split": "train", "key": labeled_key, "boxes": [box]},
                ],
            }
            write_json_atomic(ann.proposal_root / f"{ds_id}.json", proposal)

            res = ann.accept_proposals(ds_id, allow_overwrite=False)
            self.assertEqual(res["written"], 1)
            self.assertEqual(res["skipped_existing"], 1)
            written = parse_label_file(label_empty)[0]
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0].class_id, 1)
            self.assertIsNone(ann.get_proposals(ds_id))


class TestMode1AndMode2AndAnnotator(unittest.TestCase):
    def test_mode1_images_only_and_interactive_save(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            config = AppConfig.from_env(workspace_override=t / "ws")
            workspace = WorkspaceManager(config.workspace_root)
            datasets = DatasetService(config, workspace)
            ann = AnnotationService(workspace, datasets, build_default_registry())

            img1 = t / "part_a.jpg"
            img2 = t / "part_b.png"
            Image.new("RGB", (200, 100), (10, 20, 30)).save(img1)
            Image.new("RGB", (200, 100), (40, 50, 60)).save(img2)

            report = datasets.ingest_images_only([img1, img2], dataset_name="raw_parts", class_names=["defect"])
            self.assertEqual(report.total_images, 2)
            self.assertEqual(report.class_names, {0: "defect"})

            loaded = ann.load_annotator_value(report.dataset_id, "train", "part_a")
            self.assertEqual(loaded["label_list"], ["defect"])
            self.assertEqual(loaded["box_count"], 0)

            # Draw 2 boxes, including a brand-new class "scratch"
            res = ann.save_annotator_value(
                report.dataset_id,
                "train",
                "part_a",
                {
                    "image": loaded["value"]["image"],
                    "boxes": [
                        {"xmin": 10, "ymin": 10, "xmax": 100, "ymax": 60, "label": "defect"},
                        {"xmin": 120, "ymin": 20, "xmax": 180, "ymax": 80, "label": "scratch"},
                    ],
                },
            )
            self.assertEqual(res["box_count"], 2)
            self.assertEqual(res["added_classes"], ["scratch"])

            reloaded = ann.load_annotator_value(report.dataset_id, "train", "part_a")
            self.assertEqual(reloaded["box_count"], 2)
            self.assertEqual(reloaded["label_list"], ["defect", "scratch"])

    def test_mode2_custom_folder_mapping(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            config = AppConfig.from_env(workspace_override=t / "ws")
            workspace = WorkspaceManager(config.workspace_root)
            datasets = DatasetService(config, workspace)

            # Build a non-standard ZIP: my_photos/ + my_yolo_txts/ (no data.yaml)
            zip_path = t / "custom.zip"
            img_tmp = t / "sample.jpg"
            Image.new("RGB", (128, 128), (100, 100, 100)).save(img_tmp)
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(img_tmp, arcname="my_photos/item1.jpg")
                zf.writestr("my_yolo_txts/item1.txt", "0 0.5 0.5 0.4 0.4\n")
                zf.writestr("classes.txt", "bolt\nnut\n")

            staged = datasets.stage_custom_upload([zip_path])
            self.assertEqual(staged["suggested_classes"], ["bolt", "nut"])
            self.assertTrue(any(d == "my_photos" for d, _ in staged["image_dirs"]))
            self.assertTrue(any(d == "my_yolo_txts" for d, _ in staged["label_dirs"]))

            prev = datasets.preview_custom_mapping(staged["stage_id"], staged["root_dir"], "my_photos", "my_yolo_txts")
            self.assertEqual(prev["matched"], 1)

            report = datasets.commit_custom_mapping(
                stage_id=staged["stage_id"],
                root_dir=staged["root_dir"],
                dataset_name="bolts",
                image_rel_dir="my_photos",
                label_rel_dir="my_yolo_txts",
                class_names=["bolt", "nut"],
            )
            self.assertEqual(report.total_images, 1)
            self.assertEqual(report.total_labels, 1)
            self.assertEqual(report.class_names, {0: "bolt", 1: "nut"})


class TestAnnotateApiRoutes(unittest.TestCase):
    def test_api_draw_add_class_save_and_training_valid(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from application.app_controller import AppController
        from ui.annotate_api import create_annotate_router

        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig.from_env(workspace_override=Path(tmp) / "ws")
            controller = AppController(config=config)
            try:
                report = controller.dataset_service.create_demo_dataset(train_count=2, val_count=1)
                app = FastAPI()
                app.include_router(create_annotate_router(controller))
                client = TestClient(app)

                self.assertEqual(client.get("/annotate").status_code, 200)
                ds_resp = client.get("/api/annotate/datasets").json()
                self.assertEqual(len(ds_resp["datasets"]), 1)

                imgs = client.get(f"/api/annotate/dataset/{report.dataset_id}/images?split=train").json()["images"]
                self.assertEqual(len(imgs), 2)
                key = imgs[0]["key"]

                # Add a new class directly via API (syncs with data.yaml)
                cls_res = client.post(
                    f"/api/annotate/dataset/{report.dataset_id}/classes",
                    json={"name": "defect"},
                )
                self.assertEqual(cls_res.status_code, 200)
                self.assertTrue(any(c["name"] == "defect" for c in cls_res.json()["classes"]))

                # Save a newly drawn box with the new class
                save_res = client.post(
                    f"/api/annotate/dataset/{report.dataset_id}/labels/train/{key}",
                    json={"boxes": [{"label": "defect", "xmin": 10, "ymin": 10, "xmax": 100, "ymax": 100}]},
                )
                self.assertEqual(save_res.status_code, 200)
                self.assertEqual(save_res.json()["box_count"], 1)

                # Verify dataset still passes YOLO validation for training
                revalidated = controller.dataset_service.revalidate_dataset(report.dataset_id)
                self.assertTrue(revalidated.is_valid, [e.message for e in revalidated.errors])
                self.assertIn("defect", revalidated.class_names.values())
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()
