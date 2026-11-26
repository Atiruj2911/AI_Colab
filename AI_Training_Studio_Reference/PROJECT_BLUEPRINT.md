# AI Training Studio — Project Blueprint

เอกสารนี้เป็นข้อกำหนดภาพรวมสำหรับพัฒนา AI Training Studio บน Google Colab โดยใช้ Gradio เป็น Web UI และ Ultralytics YOLO เป็น Training Engine  
เป้าหมายคือทำ **End-to-End MVP ให้ใช้งานได้ก่อน** ตั้งแต่อัปโหลด Dataset → ตั้งค่า → Train → Monitor → Test → Export แล้วจึงเพิ่มความสามารถเฉพาะทางในเฟสต่อไป

> เอกสารนี้เป็น Product/Architecture Blueprint สำหรับ Cursor ไม่ใช่คำสั่งให้สร้างทุกฟีเจอร์ในครั้งเดียว ให้ทำตาม Phase และตรวจสอบผลลัพธ์ทุก Phase

## 1. Product goal

สร้างเครื่องมือสำหรับผู้ใช้ที่ต้องการ Train โมเดล YOLO ผ่านหน้าเว็บใน Google Colab โดยไม่ต้องแก้ไข Python code เพื่อทำงานปกติ

หลักการ:
- UI ใช้งานง่าย มี workflow ชัดเจน และใช้ธีม Dark/Navy เป็นค่าเริ่มต้น
- Phase 1 ต้องทำงานได้ครบวงจรจริง ไม่แบ่งเฟสแรกเป็นเพียงหน้า Upload หรือ Annotation
- Backend ต้องแยกเป็นโมดูล ไม่ผูก business logic ไว้ใน Gradio callback
- Training ต้องทำงานแยกจาก browser/UI เพื่อให้ UI reconnect ได้หาก browser หลุด ขณะที่ Colab Runtime ยังมีชีวิต
- ใช้ Colab temporary storage เป็นค่าเริ่มต้น ไม่บังคับใช้ Google Drive
- ห้ามแสดงว่ากู้คืนได้ หาก Runtime ถูกยุติและไม่มี checkpoint ภายนอกเหลืออยู่
- เพิ่ม model/task อื่นได้ภายหลังโดยไม่ต้องรื้อ application architecture

## 2. Initial technical decisions

| Area | Initial choice |
|---|---|
| Platform | Google Colab |
| UI | Gradio Blocks |
| Language | Python |
| AI framework | PyTorch |
| Initial model | Ultralytics YOLO11 |
| Initial task | Object Detection |
| Dataset format | YOLO Detection dataset |
| Workspace | Colab local temporary disk |
| Persistence | Session/config/checkpoints in runtime + explicit download to user's computer |
| Export in MVP | `.pt` and `.onnx` |
| UI theme | Dark navy / indigo |
| Architecture | Modular application + model adapter |

Use compatible, pinned dependency versions where practical. At startup, print Python, PyTorch, CUDA, GPU, Ultralytics, and Gradio versions. Do not silently upgrade packages during an active session.

## 3. User workflow

1. Open the Colab notebook and run the bootstrap/launch cell.
2. Open the Gradio application.
3. Upload a YOLO dataset ZIP or supported dataset files from the local computer.
4. Validate dataset structure, class names, labels, and split configuration.
5. Select YOLO11 model size and pretrained weights.
6. Configure essential training settings.
7. Start training and monitor progress.
8. Stop training safely when needed.
9. Test the resulting model on an uploaded image.
10. Export and download the selected model format and configuration.

The first MVP may require labels to already exist. Built-in manual annotation and AI-assisted annotation are planned for Phase 2.

## 4. Development roadmap

### Phase 1 — End-to-End MVP

**Definition of done:** A user can upload an already-labelled YOLO Detection dataset, train YOLO11, monitor training, test a model, and download/export it without editing Python code.

Deliver:
- Colab notebook/bootstrap entry point that installs/checks dependencies and launches the app.
- Gradio application shell with navigation and consistent page layout.
- Dataset upload: ZIP and supported folder/file upload flow; extract into workspace; show dataset summary.
- Dataset validation: `data.yaml`, image/label paths, class names, empty/malformed labels, train/val availability, readable images.
- Training setup: YOLO11 Detection, n/s/m model sizes, pretrained weight selection, epochs, batch, image size, device selection where available, and a small set of safe core parameters.
- Training engine: start/stop, safe subprocess/thread lifecycle, status reporting, epoch metrics, error reporting.
- Monitor: current epoch, progress, elapsed time, losses, precision, recall, mAP50, mAP50-95, GPU/VRAM where available, and training log.
- Checkpoints: preserve `best.pt`, `last.pt`, run metadata, and configuration in the runtime workspace.
- Model testing: upload one image, run inference, display annotated result and confidence controls.
- Export: download `.pt`; support `.onnx` export when dependency/runtime conditions permit; surface clear errors.
- Session basics: session ID, status, current run, configuration, and latest metrics.
- Download controls for important outputs and checkpoint files.
- Basic validation, error handling, and a short in-app help section.

Do not attempt to build a full annotation editor, advanced augmentation editor, or every export backend in this phase.

### Phase 2 — Dataset & Annotation

- Dataset browser/gallery with filtering, preview, add/remove/rename, and class management.
- Manual bounding-box editor: draw, select, move, resize, delete, change class, zoom/pan, undo/redo.
- AI Auto Annotation using a selected YOLO model and confidence threshold.
- Preview/review predictions before accepting them as labels.
- Import/export YOLO labels.
- Never overwrite existing labels without explicit confirmation.
- Dataset diagnostics: duplicates, corrupt images, missing labels, invalid classes, and class distribution.
- Train/validation/test split UI and reproducible random seed.
- Split data before generating offline augmented copies to prevent data leakage.

### Phase 3 — Augmentation

- Separate Offline Augmentation from training-time/on-the-fly augmentation.
- Image transforms: brightness, contrast, saturation, hue, blur, noise, rotation, flip, scale, crop.
- Detection transforms: mosaic, mixup, copy-paste where supported by the selected task/model.
- Probability/intensity controls and preview.
- Clearly show whether an operation creates new files or is applied during training.
- Validate that image transformations and bounding-box coordinates remain aligned.

### Phase 4 — Model & Training Expansion

- Add YOLOv8 and additional compatible YOLO versions through model adapters/registry.
- Add supported tasks such as segmentation and classification with task-specific dataset validation and UI.
- Advanced settings: optimizer, learning rate, scheduler, weight decay, momentum, warmup, freeze layers, AMP, workers, cache, and other supported options.
- Distinguish **Resume** (continue a prior run with its checkpoint/training state) from **Fine-tune** (start a new run from pretrained weights).
- Fixed-epoch and long-running/continuous mode with explicit stop controls.
- Configurable auto-save rules based on best fitness/mAP, metric threshold, epoch interval, or time interval.
- Allow safe changes to auto-save policy while training is active.
- Keep best-model and latest-resume-checkpoint semantics separate.

### Phase 5 — Advanced Tools & UX

- Training history, session browser, and experiment comparison.
- Video and webcam inference where the Colab/browser environment supports it.
- Additional export formats such as OpenVINO, TensorRT, TorchScript, and CoreML only when supported by the model, task, dependencies, and runtime.
- Recovery manager with clear checkpoint age/location and dataset availability.
- Workspace/storage usage display and cleanup tools with confirmation.
- Import/export project configuration.
- UI polish, accessibility, progress feedback, and more complete documentation.

## 5. Architecture

Keep these layers separate:

### UI layer
Gradio Blocks pages and reusable components. UI should collect user input, render status, and call application services. It must not contain training implementation logic.

### Application/controller layer
Validates requests, manages session IDs, coordinates dataset/training/inference/export services, and returns user-friendly results.

### Core services
- `DatasetService`: upload, extraction, inspection, validation, split, dataset metadata.
- `TrainingService`: create/start/stop training jobs and report state.
- `InferenceService`: load selected weights and run image inference.
- `ExportService`: export model formats and validate prerequisites.
- `CheckpointService`: save/list/download checkpoint artifacts.
- `SessionService`: persist session metadata and restore UI view after reconnect.
- Later: `AnnotationService`, `AugmentationService`, `ExperimentService`.

### Model adapter layer
Use a model registry/adapter boundary so UI and application services do not directly depend on YOLO11-specific calls. The first adapter can support YOLO11 Detection only. Future adapters/tasks should implement a consistent interface.

### Runtime/workspace layer
All runtime files should live under one configurable workspace root. Avoid scattering temporary files across the repository.

## 6. Suggested project structure

```text
ai-training-studio/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── AI_Training_Studio_Colab.ipynb
├── app.py
├── config/
│   ├── app_config.py
│   └── defaults.py
├── ui/
│   ├── app_ui.py
│   ├── components/
│   └── pages/
│       ├── home.py
│       ├── dataset.py
│       ├── training.py
│       ├── monitor.py
│       ├── testing.py
│       ├── export.py
│       ├── sessions.py
│       └── settings.py
├── application/
│   ├── app_controller.py
│   └── services/
│       ├── dataset_service.py
│       ├── training_service.py
│       ├── inference_service.py
│       ├── export_service.py
│       ├── checkpoint_service.py
│       └── session_service.py
├── core/
│   ├── training/
│   ├── dataset/
│   ├── inference/
│   ├── export/
│   └── runtime/
├── models/
│   ├── registry.py
│   ├── base_adapter.py
│   └── yolo/
│       └── yolo_adapter.py
├── infrastructure/
│   ├── storage/
│   ├── process/
│   └── monitoring/
├── utils/
│   ├── validation.py
│   ├── file_utils.py
│   └── gpu_utils.py
├── tests/
└── workspace/                 # runtime-generated; do not commit
    ├── uploads/
    ├── datasets/
    ├── weights/
    ├── runs/
    ├── exports/
    ├── sessions/
    └── temp/
```

This is a target structure, not a requirement to create empty files for every future feature. Create modules when their responsibilities are implemented, but preserve the boundaries.

## 7. Runtime, disconnect, and recovery requirements

This is a critical architectural requirement.

- Do not run a long training loop inside a browser request whose lifetime is tied to a Gradio callback.
- Training must be managed by a background worker/process with an explicit job/session ID.
- Persist job state and latest metrics to runtime files (for example JSON/JSONL) so the UI can reload them after reconnect.
- UI reconnect must attach to the existing job/session; it must not start a duplicate training job.
- Provide status/heartbeat checks, reconnect action, and a clear disconnected/running/stopped/failed state.
- Handle graceful stop and process cleanup.
- Keep `best.pt`, `last.pt`, training arguments, dataset YAML, class names, package/runtime metadata, and logs together for a run.
- Offer explicit downloads of checkpoints and session artifacts to the user's computer. Browser download behavior must be tested in Colab; do not promise silent or guaranteed background downloads.
- Since the chosen default is temporary storage and no Drive, a terminated Colab runtime destroys its local worker and files. Reconnection cannot revive a terminated runtime. Recovery is possible only from a checkpoint that was successfully downloaded or otherwise stored outside that runtime.
- Explain this limitation in the UI before training and in the recovery page.
- Do not claim that the worker can bypass Colab idle timeouts, maximum runtime limits, quota, or GPU availability constraints.

Use `last.pt` for resuming a compatible training run and `best.pt` as a starting point for a new fine-tuning run. Verify the exact Ultralytics resume behavior against the pinned version.

## 8. Data and training safety

- Validate archive contents and prevent unsafe path traversal during ZIP extraction.
- Set reasonable upload/file-size limits and show clear messages.
- Do not silently discard uploaded files.
- Preserve original uploaded dataset where practical; perform transformations on copies.
- Validate class IDs against `names` in `data.yaml`.
- Validate train/val paths and image-label pairing.
- Avoid data leakage: split before offline augmentation.
- Record random seeds and training parameters.
- Make destructive actions explicit and confirm before deleting datasets, runs, or checkpoints.
- Never overwrite existing labels or checkpoints without an explicit policy/confirmation.

## 9. UI design reference

See `ui_reference/` for the generated visual references. These are design direction and page-layout examples, not pixel-perfect requirements. Prioritize functional, readable controls over copying decorative details.

Pages:
1. `01_home_dashboard.png` — overview, quick start, recent project/run summary.
2. `02_dataset_manager.png` — upload, dataset structure, classes, validation status, image preview.
3. `03_training_setup.png` — model/weights, training mode, core hyperparameters, start action.
4. `04_training_monitor.png` — epoch progress, metric cards, charts, checkpoint and runtime status, stop action.
5. `05_model_testing.png` — model selection, confidence/IoU controls, image preview with detections.
6. `06_model_export.png` — export format selection, export status, artifact download list.
7. `07_sessions_recovery.png` — current session, history, recovery options, checkpoint availability.
8. `08_settings_autosave.png` — auto-save policies and application/runtime preferences.

### UI consistency
- Dark navy background, slightly lighter cards, restrained indigo/purple accent.
- Sidebar navigation with clear active state.
- Consistent spacing, typography, form labels, button hierarchy, and status indicators.
- Use responsive layouts suitable for Colab/browser viewport.
- Do not make the interface depend on color alone to communicate status.
- Keep the first-run workflow obvious: Dataset → Training → Monitor → Test → Export.

## 10. Testing and acceptance criteria

For each phase:
- Add unit tests for validation and core logic where practical.
- Add integration/smoke tests for the actual user workflow.
- Test with a small sample dataset before large training.
- Test missing/corrupt labels, invalid YAML, unsupported options, and missing GPU.
- Test start/stop, worker failure, UI refresh/reconnect, and duplicate-start prevention.
- Test download/export files can be opened by the intended tools.
- Show actionable error messages; log technical details separately.

Phase 1 acceptance checklist:
- [ ] Launches from the provided Colab notebook.
- [ ] User can upload a YOLO Detection dataset from their computer.
- [ ] Dataset validation reports actionable errors and warnings.
- [ ] User can select YOLO11 pretrained weights and core training settings.
- [ ] Training starts and continues independently of UI refresh/disconnect while runtime remains alive.
- [ ] Monitor displays live epoch and core metrics.
- [ ] User can request a safe stop.
- [ ] Best and last checkpoints are saved in the run workspace.
- [ ] UI can reconnect to the same live job without duplicating it.
- [ ] User can test a model on an uploaded image.
- [ ] User can download `.pt` and export/download `.onnx` when supported.
- [ ] Runtime-loss limitation is clearly communicated.

## 11. Instructions for Cursor

1. First inspect this blueprint and propose a concise implementation plan for Phase 1.
2. Inspect the current repository before creating or overwriting files.
3. Implement Phase 1 only. Do not start Phase 2 features unless required for a stable MVP.
4. Build a runnable vertical slice early: launch UI → upload/validate dataset → train → monitor → test → export.
5. Keep modules small and responsibilities clear. Avoid one giant `app.py`.
6. Do not create fake UI controls that do nothing. If a feature is not implemented, label it as planned/disabled.
7. Do not fabricate training metrics, GPU status, export success, or recovery state.
8. Use real Ultralytics APIs compatible with the pinned version; verify parameters and resume semantics.
9. Add a small demo/test dataset or clear instructions for obtaining one; do not bundle large copyrighted datasets.
10. Keep secrets and credentials out of source code.
11. Document how to launch, upload data, train, recover, and export.
12. At the end of Phase 1, report files changed, implemented features, tests performed, known limitations, and the exact next steps for Phase 2.

## 12. Definition of success

The first release is successful when a user can complete the entire basic YOLO11 Detection training workflow in Colab through the UI, without editing Python code, and can download a usable model artifact. Future phases should enhance the same application rather than replace it.
