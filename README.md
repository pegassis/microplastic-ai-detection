# Microplastic Detector

Upload a Nile Red fluorescence microscope image; YOLO11 finds particles and labels each as **ABS, Nylon, PE, PET, PS or PVC**.

```
dataset zips ─► training/prepare_dataset.py ─► training/train.py ─► models/best.pt
                                                                        │
                    React UI  ◄──── FastAPI (backend/main.py) ◄─────────┘
                                        └──► MongoDB (optional history)
```

## Status

| Part | State |
|---|---|
| Dataset preparation | Run end to end on the real dataset (2,564 images, 0 label problems) |
| Training / evaluation scripts | Written, syntax-checked, **not run** (needs your GPU) |
| FastAPI backend | Written, syntax-checked, **not run** |
| React frontend | Written, **not built or run** |

No trained model ships with this project. You create `models/best.pt` in step 4.

## 1. Get the dataset

```powershell
cd D:\Projects\microplastic-ai-detection
git clone https://github.com/sombsuk/dataset_microplastic.git ..\dataset_microplastic_raw
```

Licence: **CC BY-NC-ND 4.0** (Rermborirak et al., 2025, doi:10.1016/j.hazadv.2025.100787). Non-commercial use only, cite the paper, and do not publish the dataset or copies of it. `.gitignore` already keeps `dataset/` out of git.

## 2. Python environment

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch with CUDA **before** anything else. A plain `pip install ultralytics` on Windows pulls a CPU-only PyTorch and training will be very slow. Get the exact command for your CUDA version from https://pytorch.org/get-started/locally/ (pick Pip / Windows / CUDA), run it, then:

```powershell
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # must print True and your RTX 3050
```

## 3. Prepare the dataset

```powershell
python training\prepare_dataset.py --repo ..\dataset_microplastic_raw --out dataset
```

It extracts the zips, matches labels, drops bad label lines and duplicate images, and writes `dataset/` + `data.yaml` + `manifest.csv`. Needs about 1 GB scratch space (`--work D:\tmp` to choose where).

**Split.** All images in one zip come from one capture session and look alike, so a random split would leak near-copies into val/test and inflate scores. The script keeps whole sessions together. Result on this dataset:

| Class | train img | val img | test img |
|---|---|---|---|
| ABS | 301 | 55 | 50 |
| Nylon | 392 | 100 | 100 |
| PE | 328 | 80 | 88 |
| PET | 94 | 88 | 90 |
| PS | 167 | 52 | 53 |
| PVC | 361 | 60 | 105 |

Two weak spots to remember when you read results:
- **PET has only 94 training images (186 boxes)**, far less than the others. Expect it to be the weakest class.
- **PS has just two sessions**, so its val and test images are two halves of the same session. Its scores are less independent than the other classes.

## 4. Train and evaluate

```powershell
python training\train.py --model yolo11s.pt --imgsz 640 --epochs 100 --batch 16
```

On 6 GB VRAM, if you get "CUDA out of memory", use `--batch 8`, then `--batch 4`, or `--model yolo11n.pt`. If the data loader errors on Windows, add `--workers 2`. Smoke test first: `--epochs 3 --fraction 0.1`. The best weights are copied to `models\best.pt`.

```powershell
python training\evaluate.py     # scores the untouched TEST split -> results\test_metrics.json
```

Run the test evaluation once, when you are done tuning. Report the per-class numbers, not only the overall mAP.

## 5. Run the app

Backend (project root, venv active):

```powershell
uvicorn backend.main:app --port 8000
```

Frontend (second terminal; needs Node 18+):

```powershell
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Open http://127.0.0.1:8000/api/health first. `"model_loaded": true` means it is ready. If not, the `error` field says why.

**History (optional).** Start MongoDB and point the backend at it; without it everything works, just no history table:

```powershell
docker run -d -p 27017:27017 --name mongo mongo:7
$env:MONGO_URL = "mongodb://localhost:27017"
uvicorn backend.main:app --port 8000
```

**Single-server deployment.** `npm run build` creates `frontend/dist`; the backend serves it automatically, so one `uvicorn` process serves UI and API. Settings (environment variables): `MODEL_PATH`, `MODEL_IMGSZ` (must match the training `--imgsz`), `MAX_UPLOAD_MB`, `CORS_ORIGINS`, `MONGO_URL`, `FRONTEND_DIST`.

## API

`POST /api/detect` — multipart form: `file`, optional `conf` (0.01-0.99, default 0.25), `iou` (default 0.7). Returns detections (class, confidence, box), counts per polymer, mean confidence per polymer, and the annotated image as a base64 JPEG. `GET /api/health`, `GET /api/history`.

## Limits to state when you present this

- The model has only seen **Nile Red fluorescence images of six polymers from one lab's setup**. Other stains, microscopes, lighting, real environmental samples (water, sediment, tissue) and other polymers are untested. Expect worse results there until you add matching data.
- Each training image contains a single polymer type, so the model could learn session cues (brightness, background) instead of the polymer. That is why the split is by session. Test it on a few images from a different microscope setup before claiming it generalises.
- Polymer identity from fluorescence appearance is a screening result. Confirm important findings with FTIR or Raman.
