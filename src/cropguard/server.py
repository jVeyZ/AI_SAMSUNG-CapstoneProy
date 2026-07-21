"""
CropGuard — FastAPI backend for the Android app.

Endpoints:
    GET  /health                          → service status
    GET  /crops                           → crops and their canonical class names
    POST /predict  (multipart: file, crop) → disease prediction + probabilities
    GET  /treatment/{crop}/{disease}?lang= → static treatment (en/es/va)
    POST /chat     {crop, disease, question, lang} → AI follow-up (Gemini free tier)

Run:  uvicorn server:app --port 8000
Models are read from CROPGUARD_MODELS_DIR (default: ./models) — the env override
lets tests inject lightweight random-weight models.
"""
import os, io

import torch
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from cropguard.crop_config import CROP_CLASSES, get_num_classes, get_disease_name

from cropguard.model_def import build_model, EVAL_TRANSFORM

import cropguard.llm_advice as llm_advice
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _models_dir():
    return os.environ.get("CROPGUARD_MODELS_DIR", os.path.join(_REPO_ROOT, "models"))


app = FastAPI(title="CropGuard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # localhost tech demo only
    allow_methods=["*"],
    allow_headers=["*"],
)

_models = {}


def get_model(crop):
    """Load (and cache) the fine-tuned model for a crop, on CPU."""
    if crop not in _models:
        path = os.path.join(_models_dir(), f"cropguard_{crop.lower()}_model.pth")
        if not os.path.exists(path):
            raise HTTPException(status_code=503, detail=f"Model for {crop} not available: {path}")
        model = build_model(get_num_classes(crop), weights=None)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        _models[crop] = model
    return _models[crop]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/crops")
def crops():
    return {
        "crops": {
            crop: {"num_classes": len(classes), "classes": classes}
            for crop, classes in CROP_CLASSES.items()
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), crop: str = Form(...)):
    if crop not in CROP_CLASSES:
        raise HTTPException(status_code=400, detail=f"Unknown crop '{crop}'. Valid: {list(CROP_CLASSES)}")

    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    model = get_model(crop)
    tensor = EVAL_TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].numpy()

    pred_class = int(probs.argmax())
    return {
        "crop": crop,
        "disease": get_disease_name(crop, pred_class),
        "confidence": float(probs[pred_class]),
        "probabilities": {
            get_disease_name(crop, i): float(probs[i]) for i in range(len(probs))
        },
    }


@app.get("/treatment/{crop}/{disease}")
def treatment(crop: str, disease: str, lang: str = "en"):
    if lang not in llm_advice.VALID_LANGS:
        raise HTTPException(status_code=400, detail=f"Invalid lang '{lang}'. Valid: {list(llm_advice.VALID_LANGS)}")
    data = llm_advice.get_static_treatment(crop, disease, lang)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No treatment stored for {crop} / {disease}")
    return {"crop": crop, "disease": disease, "lang": lang, **data}


class ChatRequest(BaseModel):
    crop: str = Field(..., examples=["Rice"])
    disease: str = Field(..., examples=["Rice Blast"])
    question: str = Field(..., examples=["Can I use neem oil?"])
    lang: str = Field("en", examples=["en"])
    provider: str | None = Field(None, examples=["opencode"], description="AI provider: gemini or opencode")


@app.post("/chat")
def chat(req: ChatRequest):
    if req.crop not in CROP_CLASSES:
        raise HTTPException(status_code=400, detail=f"Unknown crop '{req.crop}'.")
    if req.disease not in CROP_CLASSES[req.crop]:
        raise HTTPException(status_code=400, detail=f"Unknown disease '{req.disease}' for {req.crop}.")
    if req.lang not in llm_advice.VALID_LANGS:
        raise HTTPException(status_code=400, detail=f"Invalid lang '{req.lang}'.")
    return llm_advice.ask_followup(req.crop, req.disease, req.question, req.lang, provider=req.provider)
