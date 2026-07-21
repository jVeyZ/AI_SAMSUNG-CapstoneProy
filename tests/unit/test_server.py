"""Unit tests for the FastAPI backend (random-weight models, no LLM key)."""
import os

import pytest
from fastapi.testclient import TestClient

from cropguard.crop_config import CROP_CLASSES


@pytest.fixture(scope="module")
def client(random_models_dir):
    os.environ["CROPGUARD_MODELS_DIR"] = random_models_dir
    os.environ.pop("GEMINI_API_KEY", None)  # force the AI-unavailable fallback path
    import cropguard.llm_advice as llm_advice
    llm_advice._gemini_client = None  # ensure cached client is not used

    import cropguard.server as server
    server._models.clear()
    with TestClient(server.app) as c:
        yield c
    os.environ.pop("CROPGUARD_MODELS_DIR", None)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_crops(client):
    r = client.get("/crops")
    assert r.status_code == 200
    crops = r.json()["crops"]
    assert set(crops) == set(CROP_CLASSES)
    for crop, info in crops.items():
        assert info["num_classes"] == len(info["classes"]) == len(CROP_CLASSES[crop])


@pytest.mark.parametrize("lang", ["en", "es", "va"])
def test_treatment_languages(client, lang):
    r = client.get("/treatment/Rice/Rice Blast", params={"lang": lang})
    assert r.status_code == 200
    body = r.json()
    assert body["lang"] == lang
    assert body["explanation"].strip()
    assert body["symptoms"] and body["treatment"] and body["prevention"]


def test_treatment_not_found(client):
    assert client.get("/treatment/Rice/Not%20A%20Disease").status_code == 404


def test_treatment_bad_lang(client):
    assert client.get("/treatment/Rice/Rice Blast", params={"lang": "xx"}).status_code == 400


@pytest.mark.parametrize("crop", list(CROP_CLASSES))
def test_predict(client, tiny_image_bytes, crop):
    r = client.post("/predict", data={"crop": crop},
                    files={"file": ("leaf.png", tiny_image_bytes, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["disease"] in CROP_CLASSES[crop]
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"]) == set(CROP_CLASSES[crop])
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_bad_crop(client, tiny_image_bytes):
    r = client.post("/predict", data={"crop": "Banana"},
                    files={"file": ("leaf.png", tiny_image_bytes, "image/png")})
    assert r.status_code == 400


def test_predict_bad_image(client):
    r = client.post("/predict", data={"crop": "Rice"},
                    files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_chat_fallback_without_key(client, monkeypatch):
    import cropguard.llm_advice as _llm
    def _fake_opencode(crop, disease, question, lang):
        raise RuntimeError("No API key")
    monkeypatch.setattr(_llm, "_ask_opencode", _fake_opencode)
    r = client.post("/chat", json={
        "crop": "Tomato", "disease": "Late Blight",
        "question": "Can I use copper?", "lang": "en", "provider": "opencode"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] is None
    assert body["note"]
    assert body["fallback"] and body["fallback"]["explanation"]


def test_chat_validation(client):
    assert client.post("/chat", json={"crop": "Banana", "disease": "X", "question": "q"}).status_code == 400
    assert client.post("/chat", json={"crop": "Rice", "disease": "Canker", "question": "q"}).status_code == 400
    assert client.post("/chat", json={"crop": "Rice", "disease": "Tungro", "question": "q", "lang": "xx"}).status_code == 400
