"""End-to-end test: real uvicorn server + full diagnosis flow over HTTP."""
import os, subprocess, sys, time

import httpx
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = "http://127.0.0.1:8123"


@pytest.fixture(scope="module")
def live_server(random_models_dir):
    env = dict(os.environ)
    env["CROPGUARD_MODELS_DIR"] = random_models_dir
    env.pop("GEMINI_API_KEY", None)  # exercise the graceful AI fallback
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8123"],
        cwd=REPO_ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                if httpx.get(f"{BASE}/health", timeout=2).status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(1)
        else:
            raise RuntimeError("server did not become healthy in 90 s")
        yield BASE
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_full_diagnosis_flow(live_server, tiny_image_bytes):
    # 1. Discover crops
    crops = httpx.get(f"{live_server}/crops", timeout=10).json()["crops"]
    assert "Rice" in crops

    # 2. Send a leaf photo for diagnosis
    r = httpx.post(f"{live_server}/predict", data={"crop": "Rice"},
                   files={"file": ("leaf.png", tiny_image_bytes, "image/png")}, timeout=60)
    assert r.status_code == 200
    diagnosis = r.json()
    disease = diagnosis["disease"]
    assert disease in crops["Rice"]["classes"]
    assert 0.0 <= diagnosis["confidence"] <= 1.0

    # 3. Fetch stored treatment in Valencian
    r = httpx.get(f"{live_server}/treatment/Rice/{disease}", params={"lang": "va"}, timeout=10)
    assert r.status_code == 200
    assert r.json()["explanation"].strip()

    # 4. Ask a follow-up question — no key in this env, so we get the graceful fallback
    r = httpx.post(f"{live_server}/chat", json={
        "crop": "Rice", "disease": disease,
        "question": "Què puc fer ara mateix?", "lang": "va"}, timeout=30)
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] is None
    assert body["fallback"]["explanation"]


def test_predict_each_crop(live_server, tiny_image_bytes):
    for crop, info in httpx.get(f"{live_server}/crops", timeout=10).json()["crops"].items():
        r = httpx.post(f"{live_server}/predict", data={"crop": crop},
                       files={"file": ("leaf.png", tiny_image_bytes, "image/png")}, timeout=60)
        assert r.status_code == 200
        assert r.json()["disease"] in info["classes"]
