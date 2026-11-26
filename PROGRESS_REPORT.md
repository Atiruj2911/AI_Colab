# รายงานความคืบหน้าโครงการ — AI Training Studio

**อัปเดตล่าสุด:** 26 พฤศจิกายน 2026  
**ตำแหน่งโปรเจกต์:** `D:\MyProject_Code\AI_Colab\`  
**สถานะภาพรวม:** Phase 1 (End-to-End MVP) + Phase 2 (Dataset & Annotation + Roboflow-style Workspace Redesign) **พัฒนาและผ่าน Automated Test ครบแล้ว** (รอการทดสอบใช้งานจริงบน UI และ Google Colab GPU)

---

## 1. สรุปความคืบหน้าตามเฟส

| เฟส | ขอบเขตงาน | สถานะ | หมายเหตุ |
|---|---|---|---|
| **Phase 1** | End-to-End MVP (Upload → Validate → Train → Monitor → Test → Export) | ✅ เสร็จในส่วนโค้ดและเทสต์ | ผ่าน E2E smoke test (เทรน YOLO11n + infer + export `.pt`/`.onnx`) |
| **Phase 2** | Dataset & Annotation (2 โหมดนำเข้า, Class Manager, Split, Diagnostics, AI Auto-Label) | ✅ เสร็จในส่วนโค้ดและเทสต์ | รองรับทั้งภาพเปล่าและชุดที่มี Label (ตรงฟอร์แมต & เลือกโฟลเดอร์เอง) |
| **Phase 2+ (Redesign)** | Dedicated Roboflow-style Annotation Workspace (HTML5 Canvas เต็มพื้นที่) | ✅ เสร็จในส่วนโค้ดและเทสต์ | แยกหน้า `✏️ Annotate` ออกจาก `📁 Dataset` ชัดเจน ผ่าน API + Validation Test |
| **Phase 3** | Augmentation Editor, Fine-tune / Resume Training | ⏳ แผนถัดไป | โครงสร้าง Adapter/Service รองรับไว้แล้ว |
| **Phase 4** | Auto-Save Policy ขั้นสูง, Project Config Import/Export | ⏳ แผนถัดไป | แสดงสถานะ Planned ใน UI |
| **Phase 5** | Multi-Model / Multi-Task, TorchScript / OpenVINO / TensorRT | ⏳ แผนถัดไป | ออกแบบ Model Registry key `(model_family, task)` รองรับไว้แล้ว |

---

## 2. รายละเอียดฟีเจอร์ที่พัฒนาเสร็จแล้ว

### Phase 1 — End-to-End MVP
- **App Shell & Theme (`ui/app_ui.py`, `ui/theme.py`):** ธีม Dark Navy / Indigo เต็มความกว้างหน้าจอ, เมนูซ้ายพร้อมสถานะงานเทรนแบบ Live
- **Dataset Upload & Validation (`application/services/dataset_service.py`, `core/dataset/validator.py`):**
  - ป้องกัน Zip Slip / symlink / ขนาดไฟล์เกินกำหนด, ปฏิเสธไฟล์ที่ไม่ใช่ `.zip` (เช่น `.rar`, `.7z`) ตั้งแต่ต้น
  - ตรวจสอบ `data.yaml`, คู่ภาพ-Label, พิกัด YOLO, สร้าง `resolved_data.yaml` อัตโนมัติ
  - มี Demo Dataset generator สำหรับทดสอบได้ทันที
- **Background Training & Monitor (`core/training/worker.py`, `infrastructure/process/job_manager.py`):**
  - รันเทรน YOLO11 (`n`/`s`/`m`) แบบ detached subprocess พร้อม heartbeat, ล็อกกันกดเริ่มซ้ำ, หยุดอย่างปลอดภัย (`stop_requested.flag`), บันทึก `metrics.jsonl` และกราฟแบบเรียลไทม์
- **Testing & Export (`application/services/inference_service.py`, `export_service.py`):**
  - ทดสอบ Detect ภาพเดียวพร้อมปรับ `Confidence` / `IoU`, Export `.pt` และ `.onnx` พร้อม config/manifest
- **Sessions & Settings:** แสดงข้อมูล Runtime, GPU/VRAM, พื้นที่ดิสก์, และเครื่องมือล้าง Workspace แบบต้องยืนยัน
- **Google Colab Launcher:** `notebooks/AI_Training_Studio_Colab.ipynb` (5 cells)

### Phase 2 — Dataset Management & 2 โหมดการนำเข้า
- **โหมด 1 (มีแค่ภาพเปล่า ยังไม่มี Label):** อัปโหลดภาพหลายไฟล์หรือ `.zip` + ตั้งชื่อคลาสเริ่มต้น ระบบสร้าง `data.yaml` และโครงสร้าง YOLO ให้พร้อมเข้าไปตีกรอบทันที
- **โหมด 2 (มีภาพ + Label `.txt` แล้ว):**
  - ถ้าตรงโครงสร้าง YOLO มาตรฐาน → โหลดใช้งานได้ทันที
  - ถ้าโครงสร้างไม่ตรง → แสดงกล่อง **เลือกโฟลเดอร์ Image / Label เอง** พร้อมตัวอย่างจำนวนไฟล์ที่จับคู่ได้จริงก่อนยืนยัน
- **Dataset Split & Diagnostics:**
  - แบ่ง `Train / Val / Test` ด้วย Random Seed โดยสร้าง **Dataset ใหม่** (ไม่แก้ของเดิม), ตัดภาพเสีย/ภาพซ้ำอัตโนมัติ (ป้องกัน Data Leakage)
  - ตรวจวินิจฉัย: ภาพซ้ำ (SHA-256), ภาพเสีย, ภาพที่ยังไม่มี Label, Label กำพร้า, รูปแบบผิด, และการกระจายคลาส
  - Export / Import Label เป็น `.zip`

### Phase 2+ — Roboflow-style Annotation Workspace (Redesign)
- **แยกหน้า `✏️ Annotate` ออกจาก `📁 Dataset`** ทั้งแบบฝังเต็มพื้นที่ในแอป และเปิดเต็มจอที่ `/annotate` (`ui/static/annotator.html`, `ui/annotate_api.py`, `ui/pages/annotate.py`)
- **Layout 5 โซน เน้น Canvas ใหญ่ที่สุด:**
  - **Top Toolbar:** `Select (V)`, `Box (B)`, `Pan (H / Space+ลาก)`, `Zoom − / % / + / Fit (0)` + ล้อเมาส์, `Undo (Ctrl+Z)`, `Redo (Ctrl+Y)`, `Save (Ctrl+S)`, `Auto-Save`, และปุ่มเปิด/ปิด `✨ AI Assist`
  - **Left Sidebar:** Image Gallery พร้อม Thumbnail, ตัวกรอง (`ทั้งหมด` / `ยังไม่ตี` / `ตีแล้ว`), และป้ายสถานะ (`✓ N` / `✓ Null` / `—`)
  - **Center:** HTML5 Canvas (Crosshair, จับย้าย, Resize 8 จุด, สีแยกตามคลาส)
  - **Right Sidebar:** รายละเอียดกรอบที่เลือก (เปลี่ยนคลาส/ดูพิกัด/ลบ), **Class Manager** เชื่อมตรงกับ `data.yaml` (เพิ่ม/เปลี่ยนชื่อ/ลบคลาสที่ไม่ได้ใช้ โดยรักษา Class ID), รายการกรอบในภาพนี้, และปุ่ม `Mark as Null`
  - **Bottom Bar:** เปลี่ยนภาพ `◀ / ▶` (`←` / `→`), ชื่อไฟล์/ขนาดภาพ, และสถานะการบันทึก (`● ยังไม่ได้บันทึก` / `✓ บันทึกแล้ว`)
- **Instant Class Selection Popup:** ปล่อยเมาส์หลังวาดกรอบแล้ว Popup เด้งทันที ค้นหา/เลือกคลาสเดิม หรือพิมพ์เพิ่มคลาสใหม่ลง `data.yaml` ได้ทันที กด `Enter` ยืนยัน (`Esc` ยกเลิก) และจำคลาสล่าสุดเป็นค่าเริ่มต้น
- **AI Assist Drawer (ซ่อนไว้เป็นค่าเริ่มต้น):** รองรับทั้ง **ทำนายภาพนี้ (ทดลอง)** และ **ตีกรอบทั้งชุด (Batch)** ด้วยโมเดลเล็ก (`yolo11n.pt`, `yolo11s.pt`) หรือโมเดลที่เทรนไว้ โดยหลังรันเสร็จกรอบขึ้นบน Canvas ในหน้าเดิมทันทีเพื่อให้ตรวจ/แก้ต่อได้

---

## 3. โครงสร้างไฟล์หลักของโปรเจกต์

- `app.py` — จุดเริ่มต้นสำหรับรันแอป (`--workspace`, `--host`, `--port`, `--share`, `--no-share`)
- `application/` — `AppController` และ Services (`DatasetService`, `AnnotationService`, `TrainingService`, `InferenceService`, `ExportService`, `CheckpointService`, `SessionService`)
- `core/` — Schemas, Dataset validator/splitter/diagnostics/annotation I/O, Training worker & metrics, Predictor, Exporter
- `models/` — `BaseModelAdapter`, `ModelRegistry`, `Yolo11DetectionAdapter`
- `infrastructure/` — `WorkspaceManager`, `safe_extract_zip`, `JobManager`, `state_store`, `system_monitor`
- `ui/` — `app_ui.py`, `theme.py`, `annotate_api.py`, `static/annotator.html`, `pages/` (`home`, `dataset`, `annotate`, `training`, `monitor`, `testing`, `export`, `sessions`, `settings`)
- `tests/` — ชุดทดสอบอัตโนมัติ (`unittest`)

---

## 4. ผลการทดสอบล่าสุด

- **Automated Tests (`python -m unittest discover -s tests -v`):** **ผ่าน 30 / 30 เทสต์ (OK)**
  - `test_dataset_validation.py` — 6 เทสต์ (ผ่าน)
  - `test_upload_rejection.py` — 3 เทสต์ (ผ่าน)
  - `test_model_registry.py` — 3 เทสต์ (ผ่าน)
  - `test_job_lifecycle.py` — 2 เทสต์ (ผ่าน)
  - `test_e2e_smoke.py` — 1 เทสต์ (เทรน YOLO11n จริง 1 epoch + reconnect + infer + export `.pt`/`.onnx`, ผ่าน)
  - `test_annotation.py` — 15 เทสต์ (พิกัด, overwrite guard, undo, rename/delete, classes, diagnostics, split, export/import, accept proposals, Mode 1, Mode 2 custom folder mapping, และ FastAPI `/annotate` + `/api/annotate/...` + ตรวจซ้ำว่าผ่าน YOLO Validation สำหรับเทรน, ผ่าน)
- **Server Health Check:** ทั้ง `http://127.0.0.1:7860/` และ `http://127.0.0.1:7860/annotate` ตอบ **HTTP 200**

---

## 5. งานที่ยังค้าง / ขั้นตอนถัดไป

1. **ทดสอบใช้งานจริงโดยผู้ใช้ (User Acceptance Testing):**
   - ทดสอบโฟลว์ครบวงจรบนเบราว์เซอร์: นำเข้าภาพเปล่า (โหมด 1) → ตีกรอบ + AI Assist ใน `✏️ Annotate` → แบ่ง Train/Val → เทรน → Monitor → Testing → Export
   - ทดสอบโหมด 2 (อัปโหลดชุดที่มี Label ทั้งแบบตรงฟอร์แมตและเลือกโฟลเดอร์เอง)
2. **ทดสอบบน Google Colab (GPU):** รัน `notebooks/AI_Training_Studio_Colab.ipynb` พร้อม `--share` เพื่อตรวจการทำงานของ `/annotate` ผ่าน Gradio Share Proxy
3. **เริ่มวางแผน Phase 3:** Augmentation Editor และ Fine-tune / Resume Training เมื่อผู้ใช้ทดสอบ Phase 1–2 เรียบร้อยแล้ว
