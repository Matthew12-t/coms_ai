import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.utils.processor import predict_crowd, model_status

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL")
DEVICE_INGEST_KEY = os.getenv("DEVICE_INGEST_KEY")

app = FastAPI()


@app.get("/")
def read_root():
    return {"status": "OK", **model_status()}


@app.get("/health")
def health():
    return {"status": "ok", **model_status()}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), canteen_id: str = Form(...)):
    image_bytes = await file.read()
    head_count = predict_crowd(image_bytes)

    payload = {"canteen_id": canteen_id, "head_count": head_count}

    if not BACKEND_URL:
        return {"status": "success", "data": payload, "forwarded": False}

    headers = {}
    if DEVICE_INGEST_KEY:
        headers["X-Device-Key"] = DEVICE_INGEST_KEY

    try:
        response = requests.post(BACKEND_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"status": "success", "data": payload, "forwarded": True}
