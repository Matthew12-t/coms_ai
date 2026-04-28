import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.utils.processor import predict_crowd

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL")

app = FastAPI()


@app.get("/")
def read_root():
    return {"status": "OK"}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), canteen_id: str = Form(...)):
    image_bytes = await file.read()
    head_count = predict_crowd(image_bytes)

    payload = {"canteen_id": canteen_id, "head_count": head_count}

    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"status": "success", "data": payload}
