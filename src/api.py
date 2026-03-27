from __future__ import annotations

import json
import time
from uuid import uuid4
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

try:
    from src.prediction import predict_image, predict_images
    from src.retrain import trigger_retraining
except ModuleNotFoundError:
    from prediction import predict_image, predict_images
    from retrain import trigger_retraining

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Scene Classification Pipeline API", version="1.0.0")
START_TIME = time.time()


@app.get("/health")
def health() -> dict:
    uptime_seconds = int(time.time() - START_TIME)
    return {"status": "up", "uptime_seconds": uptime_seconds}


@app.get("/metrics")
def metrics() -> JSONResponse:
    path = PROJECT_ROOT / "results" / "metrics.json"
    if not path.exists():
        return JSONResponse({"message": "metrics not available yet"}, status_code=404)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(data)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    safe_name = f"{uuid4().hex}_{file.filename}"
    file_path = UPLOAD_DIR / safe_name
    content = await file.read()
    file_path.write_bytes(content)

    try:
        result = predict_image(PROJECT_ROOT, file_path)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    return JSONResponse(result)


@app.post("/predict-bulk")
async def predict_bulk(files: list[UploadFile] = File(...)) -> JSONResponse:
    saved_paths = []
    original_names = []

    for item in files:
        safe_name = f"{uuid4().hex}_{item.filename}"
        target = UPLOAD_DIR / safe_name
        target.write_bytes(await item.read())
        saved_paths.append(target)
        original_names.append(item.filename)

    try:
        preds = predict_images(PROJECT_ROOT, saved_paths)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    results = []
    for filename, pred in zip(original_names, preds):
        results.append({"filename": filename, "prediction": pred})

    return JSONResponse({"count": len(results), "results": results})


@app.post("/upload-bulk")
async def upload_bulk(files: list[UploadFile] = File(...)) -> JSONResponse:
    saved_files = []
    for item in files:
        target = UPLOAD_DIR / item.filename
        target.write_bytes(await item.read())
        saved_files.append(item.filename)
    return JSONResponse({"uploaded_count": len(saved_files), "files": saved_files})


@app.post("/retrain")
def retrain() -> JSONResponse:
    result = trigger_retraining(PROJECT_ROOT)
    return JSONResponse({"message": "retraining complete", "result": result})
