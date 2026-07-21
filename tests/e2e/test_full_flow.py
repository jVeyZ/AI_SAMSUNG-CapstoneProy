"""End-to-end tests: real uvicorn server + full diagnosis flow over HTTP.

Tests that require real fine-tuned models (not random weights) are marked
@pytest.mark.local_only and skipped in CI.
"""
import os, subprocess, sys, time

import httpx
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = "http://127.0.0.1:8123"


@pytest.fixture(scope="module")
def live_server(random_models_dir):
    env = dict(os.environ)
    env["CROPGUARD_MODELS_DIR"] = random_models_dir
    env.pop("GEMINI_API_KEY", None)
    src_dir = os.path.join(REPO_ROOT, "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "cropguard.server:app", "--host", "127.0.0.1", "--port", "8123"],
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


# ---------------------------------------------------------------------------
# Full flow tests (random-weight models, run in CI)
# ---------------------------------------------------------------------------

class TestFullFlowPNG:
    """Complete diagnosis flow using PNG images."""

    def test_discover_crops(self, live_server):
        crops = httpx.get(f"{live_server}/crops", timeout=10).json()["crops"]
        assert set(crops.keys()) == {"Tomato", "Rice", "Orange"}

    @pytest.mark.parametrize("crop", ["Tomato", "Rice", "Orange"])
    def test_predict_crop(self, live_server, tiny_image_bytes, crop):
        r = httpx.post(f"{live_server}/predict", data={"crop": crop},
                       files={"file": ("leaf.png", tiny_image_bytes, "image/png")}, timeout=60)
        assert r.status_code == 200
        body = r.json()
        assert body["crop"] == crop
        assert body["disease"] in [d for d in body["probabilities"]]

    def test_fetch_treatment_after_predict(self, live_server, tiny_image_bytes):
        r = httpx.post(f"{live_server}/predict", data={"crop": "Rice"},
                       files={"file": ("leaf.png", tiny_image_bytes, "image/png")}, timeout=60)
        disease = r.json()["disease"]
        r2 = httpx.get(f"{live_server}/treatment/Rice/{disease}", params={"lang": "en"}, timeout=10)
        assert r2.status_code == 200
        assert r2.json()["explanation"].strip()

    def test_chat_fallback(self, live_server, tiny_image_bytes):
        r = httpx.post(f"{live_server}/predict", data={"crop": "Tomato"},
                       files={"file": ("leaf.png", tiny_image_bytes, "image/png")}, timeout=60)
        disease = r.json()["disease"]
        r2 = httpx.post(f"{live_server}/chat", json={
            "crop": "Tomato", "disease": disease,
            "question": "Can I use copper?", "lang": "en",
            "provider": "__test_only__"}, timeout=30)
        assert r2.status_code == 200
        body = r2.json()
        assert body["answer"] is None
        assert body["fallback"] and body["fallback"]["explanation"]

    def test_treatment_all_languages(self, live_server, tiny_image_bytes):
        r = httpx.post(f"{live_server}/predict", data={"crop": "Rice"},
                       files={"file": ("leaf.png", tiny_image_bytes, "image/png")}, timeout=60)
        disease = r.json()["disease"]
        for lang in ("en", "es", "va"):
            r2 = httpx.get(f"{live_server}/treatment/Rice/{disease}", params={"lang": lang}, timeout=10)
            assert r2.status_code == 200
            assert r2.json()["explanation"].strip()


class TestFullFlowJPG:
    """Complete diagnosis flow using JPG images."""

    @pytest.mark.parametrize("crop", ["Tomato", "Rice", "Orange"])
    def test_predict_crop_jpg(self, live_server, tiny_jpg_bytes, crop):
        r = httpx.post(f"{live_server}/predict", data={"crop": crop},
                       files={"file": ("leaf.jpg", tiny_jpg_bytes, "image/jpeg")}, timeout=60)
        assert r.status_code == 200
        body = r.json()
        assert body["crop"] == crop
        assert body["disease"] in [d for d in body["probabilities"]]

    def test_jpg_probabilities_sum_to_one(self, live_server, tiny_jpg_bytes):
        r = httpx.post(f"{live_server}/predict", data={"crop": "Tomato"},
                       files={"file": ("leaf.jpg", tiny_jpg_bytes, "image/jpeg")}, timeout=60)
        probs = r.json()["probabilities"]
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)

    def test_jpg_then_treatment(self, live_server, tiny_jpg_bytes):
        r = httpx.post(f"{live_server}/predict", data={"crop": "Orange"},
                       files={"file": ("fruit.jpg", tiny_jpg_bytes, "image/jpeg")}, timeout=60)
        assert r.status_code == 200
        disease = r.json()["disease"]
        r2 = httpx.get(f"{live_server}/treatment/Orange/{disease}", params={"lang": "en"}, timeout=10)
        assert r2.status_code == 200
        assert r2.json()["explanation"].strip()


class TestBadInputs:
    """Verify bad inputs are properly rejected."""

    def test_bad_crop(self, live_server, tiny_image_bytes):
        r = httpx.post(f"{live_server}/predict", data={"crop": "Banana"},
                       files={"file": ("leaf.png", tiny_image_bytes, "image/png")}, timeout=60)
        assert r.status_code == 400

    def test_bad_image_bytes(self, live_server):
        r = httpx.post(f"{live_server}/predict", data={"crop": "Rice"},
                       files={"file": ("x.png", b"garbage", "image/png")}, timeout=60)
        assert r.status_code == 400

    def test_chat_bad_crop(self, live_server):
        r = httpx.post(f"{live_server}/chat", json={
            "crop": "Banana", "disease": "X", "question": "q"}, timeout=10)
        assert r.status_code == 400

    def test_chat_bad_disease(self, live_server):
        r = httpx.post(f"{live_server}/chat", json={
            "crop": "Rice", "disease": "Canker", "question": "q"}, timeout=10)
        assert r.status_code == 400

    def test_treatment_not_found(self, live_server):
        r = httpx.get(f"{live_server}/treatment/Rice/NotADisease", timeout=10)
        assert r.status_code == 404

    def test_treatment_bad_lang(self, live_server):
        r = httpx.get(f"{live_server}/treatment/Rice/Rice Blast", params={"lang": "xx"}, timeout=10)
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Local-only tests: need real fine-tuned models + real leaf images
# ---------------------------------------------------------------------------

local_only = pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Needs real models and images (not available in CI)",
)


@local_only
class TestLocalRealModels:
    """Tests that need real fine-tuned models — run locally only."""

    @pytest.fixture(scope="class")
    def real_server(self):
        """Start server with real models from models/ directory."""
        src_dir = os.path.join(REPO_ROOT, "src")
        env = dict(os.environ)
        env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "cropguard.server:app",
             "--host", "127.0.0.1", "--port", "8124"],
            cwd=REPO_ROOT, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.time() + 90
            while time.time() < deadline:
                try:
                    if httpx.get("http://127.0.0.1:8124/health", timeout=2).status_code == 200:
                        break
                except httpx.TransportError:
                    time.sleep(1)
            else:
                pytest.skip("Real model server did not start")
            yield "http://127.0.0.1:8124"
        finally:
            proc.terminate()
            proc.wait(timeout=15)

    def test_real_model_tomato(self, real_server):
        """Upload a real leaf photo — expected high confidence on a known disease."""
        img_path = os.path.join(REPO_ROOT, "data", "Tomato", "test", "Tomato_Late_Blight")
        if not os.path.isdir(img_path):
            pytest.skip("No real Tomato test data")
        files = [f for f in os.listdir(img_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not files:
            pytest.skip("No test images")
        with open(os.path.join(img_path, files[0]), "rb") as fh:
            r = httpx.post(f"{real_server}/predict", data={"crop": "Tomato"},
                           files={"file": ("leaf.jpg", fh, "image/jpeg")}, timeout=60)
        assert r.status_code == 200
        body = r.json()
        assert body["disease"] in [d for d in body["probabilities"]]
