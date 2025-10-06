from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List
import os, uuid, glob, shutil
from deepface import DeepFace

from src.pipeline.face_detection_pipeline import FaceDetectionPipeline
from src.pipeline.face_recognition_pipeline import FaceRecognitionPipeline

app = FastAPI(title="Face Detection & Recognition API")

# Pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

det = FaceDetectionPipeline(model_path = "yolov8n-face.pt", device = device, conf = 0.25)
rec = FaceRecognitionPipeline(db_path = "img_real", model_name = "Facenet", threshold = 0.8)

DB_DIR = Path("img_real").resolve()
DB_DIR.mkdir(parents=True, exist_ok=True)

# Health

@app.get("/health")  # Fonction appelée pour répondre
def health():
    return {"status": "ok", "device": device}


# Upload vers img_real

@app.post("/upload")
async def upload_images(person: str = Form(...), files: List[UploadFile] = File(...), warmup : bool = True):  # # si True, on “réveille” DeepFace après upload (1 ligne)
    person = person.strip()
    if not person:
        return JSONResponse({"error" : "person is required"}, status_code = 400)
    
    person_dir = DB_DIR / person
    person_dir.mkdir(parents=True,exist_ok=True)

    saved = []
    for f in files:
        # lecture et décodage
        data = await f.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": f"decode failed: {f.filename}"}, status_code=400)

        # nom unique
        ext = os.path.splitext(f.filename)[1].lower() or ".jpg"
        out_path = person_dir / f"{uuid.uuid4().hex}{ext}"
        cv2.imwrite(str(out_path), img)
        saved.append(out_path.name)

    return {"person": person, "saved": saved, "count": len(saved)}

# Infer

@app.post("/infer")
async def infer(file: UploadFile = File(...), margin: int = 10): # la fonction est asynchrone (FastAPI/uvicorn peuvent gérer plusieurs requêtes en parallèle sans bloquer pendant les I/O).
    data = await file.read()  # Lit asynchroniquement le contenu binaire du fichier uploadé.
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # Convertit les bytes en np.ndarray de uint8, puis OpenCV décode cette image (BGR). Equivalent à cv2.imread, mais pour un buffer en mémoire (pas un fichier disque).
    if img is None:
        return JSONResponse({"error": "image decode failed"}, status_code=400)
    
    h, w = img.shape[:2]
    boxes = det.detect_boxes(img)
    results = []

    for (x1, y1, x2, y2) in boxes:
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        crop = img[y1:y2, x1:x2]
        name = rec.identify_one(crop)
        results.append({"box": [x1, y1, x2, y2], "name": name})

    return {"results": results}