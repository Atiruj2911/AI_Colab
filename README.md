# AI Training Studio (Phase 2 — Dataset & Annotation MVP)

เว็บแอปพลิเคชันแบบ **Gradio** สำหรับเทรนโมเดล **Ultralytics YOLO11 (Object Detection)** บน **Google Colab** (และรันทดสอบบนเครื่อง Local ได้) ครบวงจรตั้งแต่ **อัปโหลด Dataset → ตรวจสอบ → เทรน → ติดตามผล → ทดสอบ → Export/ดาวน์โหลด** โดยไม่ต้องแก้ไขโค้ด Python

---

## ความสามารถใน Phase 1 (End-to-End MVP)

- **Dataset Manager:** อัปโหลดไฟล์ `.zip` (หรือไฟล์ชุด YOLO), ป้องกัน Zip Slip / path traversal, ตรวจสอบ `data.yaml`, โครงสร้าง `train/val`, การอ่านรูปภาพ, รูปแบบพิกัด YOLO (`class cx cy w h`) และ range ของ `class_id`, แสดงโครงสร้างโฟลเดอร์และภาพตัวอย่างพร้อมกรอบ Ground Truth, มีปุ่ม **สร้าง Demo Dataset** สำหรับทดสอบระบบทันที
- **Training Setup (YOLO11 Detection):** รองรับ `YOLO11n`, `YOLO11s`, `YOLO11m` พร้อม Pretrained Weights, ตั้งค่า `Epochs`, `Batch Size`, `Image Size`, `Device`, `Seed`, `Patience` และชื่อ Run
- **Disconnect-Safe Background Training Engine:** แยก Training ไปรันใน Background Subprocess (`core/training/worker.py`) อิสระจาก Gradio callback, มี active-job lock ป้องกันการเริ่มเทรนซ้ำซ้อน, บันทึก `job_state.json` และ `metrics.jsonl` แบบ atomic พร้อม heartbeat
- **Training Monitor:** แสดง Epoch, Progress, Elapsed, ETA, Loss, Precision, Recall, mAP50, mAP50-95, GPU/VRAM จริง, กราฟ 4 แท็บ, Training Log, ปุ่ม **Reconnect** และ **Graceful Stop**
- **Checkpoints & Run Artifacts:** เก็บ `best.pt`, `last.pt`, `train_config.json`, `data.yaml`, `runtime_meta.json` และ `train.log` ไว้ด้วยกันใน `workspace/runs/<run_id>/` พร้อมปุ่มดาวน์โหลด
- **Model Testing / Detect:** เลือกโมเดลจาก Run หรืออัปโหลด `.pt`, ปรับ `Confidence` / `IoU (NMS)`, แสดงภาพผลลัพธ์พร้อม Bounding Boxes เทียบกับภาพต้นฉบับ และตารางรายการวัตถุที่ตรวจพบ
- **Model Export:** ดาวน์โหลด `.pt` และ Export เป็น `.onnx` (เมื่อสภาพแวดล้อมพร้อม) พร้อมแนบ config/manifest
- **Sessions & Settings:** แสดงสถานะเซสชันปัจจุบัน, ประวัติ Run ใน Runtime นี้, เวอร์ชันแพ็กเกจทั้งหมด, การใช้งานพื้นที่ Workspace และเครื่องมือล้างพื้นที่แบบต้องยืนยัน

## ความสามารถใน Phase 2 (Dataset & Annotation)

หน้า **Dataset → จัดการ Dataset** (เลือก Dataset แล้วใช้งานได้ 5 แท็บ):

- **แก้ไข Label (ตาราง):** เลือกภาพจาก Gallery (กรองตาม Split และสถานะ Label) แก้พิกัด `x1,y1,x2,y2` เป็น pixel และคลาสในตาราง แล้วบันทึก ระบบจะไม่เขียนทับ Label เดิมจนกว่าจะติ๊กยืนยัน และย้อนกลับการบันทึกล่าสุดได้ 1 ครั้ง นอกจากนี้ยังเปลี่ยนชื่อภาพ และลบภาพ (ต้องยืนยัน) ได้
- **AI Auto-Label:** เลือกโมเดล `.pt` ที่เทรนแล้ว ปรับ Confidence / IoU / Image Size แล้วดูผลเสนอในแกลเลอรีก่อน ระบบยังไม่เขียน Label จนกว่าจะกด **ยอมรับผลเสนอ** (ขอบเขต "ทุกภาพ" จะเขียนทับ Label เดิมได้ต่อเมื่อติ๊กยืนยัน)
- **วินิจฉัย Dataset:** ภาพซ้ำ (เทียบ SHA-256), ภาพเสีย, ภาพที่ยังไม่มี Label, Label กำพร้า, Label รูปแบบผิด, class_id เกินขอบเขต, คลาสที่ไม่มีตัวอย่าง และการกระจายคลาสต่อ Split
- **แบ่ง Train / Val / Test:** สร้าง Dataset ใหม่ด้วย Seed (ไม่แก้ไข Dataset เดิม) ตัดภาพเสียและภาพซ้ำออกได้ ควรทำก่อนทำ Augmentation เพื่อกันข้อมูลรั่วระหว่าง Train/Val
- **คลาส:** แก้ชื่อคลาส, เพิ่มคลาสใหม่, ลบคลาส (ได้เฉพาะคลาสที่ id สูงสุดและไม่มี Box ใช้งาน)
- **Export / Import Label:** Export เป็น ZIP (`data.yaml`, `classes.txt`, `labels/<split>/...`) และ Import กลับโดยจับคู่ตามชื่อไฟล์ภาพ

**ข้อจำกัดของ Phase 2 (MVP):**
- ตัวแก้ไขกรอบเป็น **ตาราง** ยังไม่มีการลาก/ย้าย/ปรับขนาดกรอบบนภาพ
- ย้อนกลับได้เพียง 1 ระดับต่อการบันทึก (ต่อภาพ)
- ลบคลาสได้เฉพาะคลาสที่ id สูงสุดและไม่มี Box ใช้งาน เพื่อไม่ให้ id ของคลาสอื่นเปลี่ยน
- Auto-Label ใช้ได้กับโมเดลที่ Adapter รองรับ `predict_boxes_batch` (ตอนนี้คือ YOLO11 Detection)

> ฟีเจอร์ของ Phase 3–5 (Augmentation Editor, Fine-tune/Resume, Auto-Save Policy ขั้นสูง, Multi-model/Task, TorchScript/OpenVINO/TensorRT) ถูกออกแบบสถาปัตยกรรมรองรับไว้แล้ว และแสดงสถานะ *Planned / Disabled* อย่างชัดเจนใน UI

---

## วิธีเปิดใช้งาน

### 1. บน Google Colab (แนะนำ)
แต่ละคนเปิดลิงก์นี้ด้วยบัญชี Colab ของตัวเอง แล้วเลือก GPU จาก **Runtime → Change runtime type** จากนั้นกด **Runtime → Run all**

https://colab.research.google.com/github/Atiruj2911/AI_Colab/blob/main/notebooks/AI_Training_Studio_Colab.ipynb

โน้ตบุ๊กจะดึงโค้ดล่าสุดจาก GitHub มาไว้บน runtime ของคนนั้น ข้อมูลเทรนอยู่คนละดิสก์ ไม่ปนกัน เมื่อมีเวอร์ชันใหม่ ให้ Restart session แล้ว Run all อีกครั้ง

### 2. บนเครื่อง Local (Windows / Linux / macOS)
```bash
pip install -r requirements.txt
python app.py --no-share
```
จากนั้นเปิดเบราว์เซอร์ไปที่ `http://127.0.0.1:7860`

ตัวเลือก CLI ที่รองรับ:
- `--workspace <path>`: กำหนดโฟลเดอร์ runtime workspace (หรือใช้ env `AI_STUDIO_WORKSPACE`)
- `--port <port>`: พอร์ตของเซิร์ฟเวอร์ (ค่าเริ่มต้น `7860`)
- `--share` / `--no-share`: เปิดหรือปิดลิงก์สาธารณะของ Gradio

---

## โครงสร้าง Dataset ที่รองรับ (YOLO Detection)

อัปโหลดเป็นไฟล์ `.zip` ที่มีโครงสร้างตัวอย่างดังนี้:

```text
my_dataset/
├── data.yaml
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── val/
│       ├── img_101.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img_001.txt
    │   └── ...
    └── val/
        ├── img_101.txt
        └── ...
```

ตัวอย่าง `data.yaml`:
```yaml
path: .
train: images/train
val: images/val
nc: 2
names:
  0: person
  1: helmet
```

แต่ละบรรทัดในไฟล์ `.txt` ของ Label:
```text
<class_id> <x_center> <y_center> <width> <height>
```
โดยค่าพิกัดทั้งหมดถูก normalize อยู่ในช่วง `0.0` ถึง `1.0`

---

## ข้อจำกัดเรื่อง Colab Temporary Storage และการกู้คืน

- ระบบใช้พื้นที่ดิสก์ชั่วคราวของ Colab Runtime เป็นค่าเริ่มต้น (ไม่บังคับเชื่อม Google Drive)
- **กรณีแท็บเบราว์เซอร์หลุด แต่ Colab Runtime ยังทำงานอยู่:** Worker ที่เทรนอยู่จะไม่หยุด และสามารถเปิดหน้าแอปแล้วกด **Reconnect** ในหน้า **Monitor** หรือ **Sessions** เพื่อดูความคืบหน้าต่อได้โดยไม่เริ่มเทรนซ้ำ
- **กรณี Colab Runtime ถูกยุติ (Idle timeout / ครบโควต้า / Restart Runtime):** ข้อมูลและ Worker ใน Runtime จะถูกลบทั้งหมด ไม่สามารถกู้คืนจากภายใน Colab ได้ — กรุณาดาวน์โหลด `best.pt` และ `last.pt` ลงเครื่องคอมพิวเตอร์เมื่อได้ผลลัพธ์ที่ต้องการ

---

## การทดสอบ (Tests)

```bash
python -m unittest discover -s tests -v
```
(หรือ `pytest -v` หากติดตั้ง `pytest` ไว้)
