"""FastAPI backend: upload a microscope image, get YOLO11 microplastic detections back.

Run:  uvicorn backend.main:app --reload --port 8000     (from the project root)
Env:  MODEL_PATH (default models/best.pt), MODEL_IMGSZ (640), MONGO_URL (optional history),
      MAX_UPLOAD_MB (15), CORS_ORIGINS (comma list), FRONTEND_DIST (built React app to serve)
"""
from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

log = logging.getLogger("microplastic")
logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "best.pt"))
IMGSZ = int(os.getenv("MODEL_IMGSZ", "640"))
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "15")) * 1024 * 1024
MAX_PIXELS = 40_000_000
MONGO_URL = os.getenv("MONGO_URL")
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", ROOT / "frontend" / "dist"))
ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

Image.MAX_IMAGE_PIXELS = MAX_PIXELS
state = {"model": None, "device": None, "mongo": None, "model_error": None}
predict_lock = threading.Lock()  # one inference at a time; the model object is not thread-safe


def load_model():
    if not MODEL_PATH.exists():
        state["model_error"] = f"Model file not found: {MODEL_PATH}. Train a model (training/train.py) or set MODEL_PATH."
        log.warning(state["model_error"])
        return
    try:
        import torch
        from ultralytics import YOLO

        state["device"] = 0 if torch.cuda.is_available() else "cpu"
        model = YOLO(str(MODEL_PATH))
        model.predict(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8), imgsz=IMGSZ, device=state["device"], verbose=False)
        state["model"] = model
        log.info("Model loaded from %s on %s", MODEL_PATH, state["device"])
    except Exception as e:  # keep the API up so /api/health can explain what is wrong
        state["model_error"] = f"Could not load model: {e}"
        log.exception("model load failed")


def connect_mongo():
    if not MONGO_URL:
        return
    try:
        from pymongo import MongoClient

        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        state["mongo"] = client[os.getenv("MONGO_DB", "microplastic")]["analyses"]
        log.info("MongoDB connected")
    except Exception as e:
        log.warning("MongoDB unavailable, history disabled: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    connect_mongo()
    yield


app = FastAPI(title="Microplastic Detection API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])


def _encode_jpeg(arr_rgb: np.ndarray, max_side: int = 1600) -> str:
    img = Image.fromarray(arr_rgb)
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/api/health")
def health():
    model = state["model"]
    return {
        "status": "ok" if model else "model_not_loaded",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "classes": list(model.names.values()) if model else [],
        "device": str(state["device"]) if model else None,
        "history_enabled": state["mongo"] is not None,
        "error": state["model_error"],
    }


@app.post("/api/detect")
def detect(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.7),
):
    model = state["model"]
    if model is None:
        raise HTTPException(503, state["model_error"] or "Model not loaded")
    if not (0.01 <= conf <= 0.99 and 0.1 <= iou <= 0.95):
        raise HTTPException(422, "conf must be 0.01-0.99 and iou 0.1-0.95")

    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_BYTES // 1024 // 1024} MB)")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "Could not read this file as an image (use JPG, PNG, BMP or TIFF).")

    t0 = time.perf_counter()
    with predict_lock:
        result = model.predict(img, conf=conf, iou=iou, imgsz=IMGSZ, device=state["device"], verbose=False, agnostic_nms=True)[0]
    ms = round((time.perf_counter() - t0) * 1000, 1)

    names = model.names
    detections = []
    for (x1, y1, x2, y2), c, k in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
        detections.append({
            "class": names[int(k)], "confidence": round(c, 4),
            "bbox": [round(v, 1) for v in (x1, y1, x2, y2)],
            "width_px": round(x2 - x1, 1), "height_px": round(y2 - y1, 1),
        })
    detections.sort(key=lambda d: -d["confidence"])

    counts = {n: 0 for n in names.values()}
    conf_sum = {n: 0.0 for n in names.values()}
    for d in detections:
        counts[d["class"]] += 1
        conf_sum[d["class"]] += d["confidence"]
    mean_conf = {n: (round(conf_sum[n] / counts[n], 3) if counts[n] else None) for n in counts}

    annotated = _encode_jpeg(np.ascontiguousarray(result.plot(line_width=2)[..., ::-1]))  # BGR -> RGB

    record = {
        "filename": file.filename, "created_at": datetime.now(timezone.utc),
        "image_size": list(img.size), "conf_threshold": conf, "iou_threshold": iou,
        "total": len(detections), "counts": counts, "mean_confidence": mean_conf, "inference_ms": ms,
    }
    analysis_id = None
    if state["mongo"] is not None:
        try:
            analysis_id = str(state["mongo"].insert_one(dict(record)).inserted_id)
        except Exception as e:
            log.warning("could not save history: %s", e)

    record["created_at"] = record["created_at"].isoformat()
    return {**record, "id": analysis_id, "detections": detections, "annotated_image": annotated,
            "model": MODEL_PATH.name}


@app.get("/api/history")
def history(limit: int = Query(20, ge=1, le=100)):
    col = state["mongo"]
    if col is None:
        return {"enabled": False, "items": []}
    items = []
    for doc in col.find().sort("created_at", -1).limit(limit):
        doc["id"] = str(doc.pop("_id"))
        doc["created_at"] = doc["created_at"].isoformat()
        items.append(doc)
    return {"enabled": True, "items": items}


# Serve the built React app (npm run build) from the same server, if present. Keep this LAST.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
