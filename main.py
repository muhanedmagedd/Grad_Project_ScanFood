import io
import base64
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from inference import load_models, predict


# ── Startup / shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()          # load YOLO + MiDaS once at server start
    yield
    # nothing to clean up


app = FastAPI(
    title="ScanFood API",
    description="Estimate calories from a food photo using YOLO + MiDaS",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response schemas ──────────────────────────────────────────────────────────
class FoodItem(BaseModel):
    name:       str
    weight_g:   float
    calories:   float


class PredictResponse(BaseModel):
    items:           List[FoodItem]
    total_weight_g: float
    total_calories:  float
    annotated_image: str          # base-64 JPEG
    error:           Optional[str] = None


# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    # ── validate ──
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image.")

    raw   = await file.read()
    nparr = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # ── run pipeline ──
    result = predict(image)

    # ── encode annotated image → base64 JPEG ──
    _, buf = cv2.imencode(".jpg", result["annotated_image"], [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf).decode()

    return PredictResponse(
        items=[FoodItem(**item) for item in result["items"]],
        total_weight_g=result["total_weight_g"],
        total_calories=result["total_calories"],
        annotated_image=img_b64,
        error=result.get("error"),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
