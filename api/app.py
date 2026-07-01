import io
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from src.inference import load_model, predict

app = FastAPI(title="Defect Detection API", version="1.0")

# Cache loaded models so we don't reload on every request
_model_cache: dict = {}
AVAILABLE_CATEGORIES = ["bottle", "cable", "capsule", "hazelnut", "metal_nut",
                        "pill", "screw", "toothbrush", "transistor", "zipper"]


def get_model(category: str):
    if category not in _model_cache:
        _model_cache[category] = load_model(category)
    return _model_cache[category]


@app.get("/categories")
def list_categories():
    return {"categories": AVAILABLE_CATEGORIES}


@app.post("/predict/{category}")
async def predict_defect(category: str, file: UploadFile = File(...)):
    if category not in AVAILABLE_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Unknown category: {category}")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Save temp file for inference function
    tmp_path = Path(f"/tmp/{file.filename}")
    image.save(tmp_path)

    try:
        model = get_model(category)
        result = predict(model, str(tmp_path))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return JSONResponse({
        "category": category,
        "anomaly_score": result["score"],
        "is_defective": bool(result["score"] > 5.5),  # adjust after calibrating threshold
    })


@app.post("/predict/{category}/heatmap")
async def predict_heatmap(category: str, file: UploadFile = File(...)):
    """Returns the annotated heatmap overlay as a JPEG image."""
    if category not in AVAILABLE_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Unknown category: {category}")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tmp_path = Path(f"/tmp/{file.filename}")
    image.save(tmp_path)

    model = get_model(category)
    result = predict(model, str(tmp_path))

    _, buffer = cv2.imencode(".jpg", result["overlay_bgr"])
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")