# Sapiens Pose Runner (YOLO + Sapiens TorchScript)

Pipeline minima per eseguire pose estimation con **Sapiens (Meta)** su video:
- Bounding box persona: **YOLOv8 (ultralytics)**
- Pose: **Sapiens TorchScript (.pt2)**

Output:
- Video overlay con skeleton (COCO-17)
- JSONL con 17 keypoints per frame

## Requisiti
- Linux / WSL Ubuntu (consigliato) o Linux nativo
- Python 3.10+
- `ffmpeg` (per eventuale estrazione/ricostruzione video)

## Installazione
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
