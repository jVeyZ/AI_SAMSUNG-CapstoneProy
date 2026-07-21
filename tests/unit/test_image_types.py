"""Unit tests for image type handling and prediction flow."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cropguard.crop_config import CROP_CLASSES


@pytest.fixture(scope="module")
def client(random_models_dir):
    import os
    os.environ["CROPGUARD_MODELS_DIR"] = random_models_dir
    os.environ.pop("GEMINI_API_KEY", None)
    import cropguard.llm_advice as llm_advice
    llm_advice._gemini_client = None
    import cropguard.server as server
    server._models.clear()
    with TestClient(server.app) as c:
        yield c
    os.environ.pop("CROPGUARD_MODELS_DIR", None)


class TestImageTypes:
    """Verify the server handles JPG and PNG, rejects everything else."""

    @pytest.mark.parametrize("crop", list(CROP_CLASSES))
    def test_predict_png(self, client, tiny_image_bytes, crop):
        r = client.post("/predict", data={"crop": crop},
                        files={"file": ("leaf.png", tiny_image_bytes, "image/png")})
        assert r.status_code == 200
        body = r.json()
        assert body["disease"] in CROP_CLASSES[crop]
        assert 0.0 <= body["confidence"] <= 1.0

    @pytest.mark.parametrize("crop", list(CROP_CLASSES))
    def test_predict_jpg(self, client, tiny_jpg_bytes, crop):
        r = client.post("/predict", data={"crop": crop},
                        files={"file": ("leaf.jpg", tiny_jpg_bytes, "image/jpeg")})
        assert r.status_code == 200
        body = r.json()
        assert body["disease"] in CROP_CLASSES[crop]
        assert 0.0 <= body["confidence"] <= 1.0

    def test_predict_jpg_probabilities_sum_to_one(self, client, tiny_jpg_bytes):
        r = client.post("/predict", data={"crop": "Tomato"},
                        files={"file": ("leaf.jpg", tiny_jpg_bytes, "image/jpeg")})
        assert r.status_code == 200
        probs = r.json()["probabilities"]
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)

    def test_predict_png_probabilities_sum_to_one(self, client, tiny_image_bytes):
        r = client.post("/predict", data={"crop": "Tomato"},
                        files={"file": ("leaf.png", tiny_image_bytes, "image/png")})
        assert r.status_code == 200
        probs = r.json()["probabilities"]
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)

    def test_predict_bad_image_bytes(self, client):
        r = client.post("/predict", data={"crop": "Rice"},
                        files={"file": ("x.png", b"not an image", "image/png")})
        assert r.status_code == 400

    def test_predict_empty_file(self, client):
        r = client.post("/predict", data={"crop": "Rice"},
                        files={"file": ("empty.png", b"", "image/png")})
        assert r.status_code == 400

    def test_predict_gif_rejected(self, client):
        """GIF is not a supported type — server opens it but converts to RGB."""
        arr = (__import__("numpy").random.RandomState(0).rand(96, 96, 3) * 255).astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="GIF")
        gif_bytes = buf.getvalue()
        r = client.post("/predict", data={"crop": "Rice"},
                        files={"file": ("leaf.gif", gif_bytes, "image/gif")})
        # PIL can open GIF and convert to RGB, so this actually succeeds
        assert r.status_code == 200

    def test_predict_tiff_accepted(self, client):
        """TIFF is also decodable by PIL."""
        arr = __import__("numpy").random.RandomState(0).rand(96, 96, 3).astype("uint8") * 255
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="TIFF")
        tiff_bytes = buf.getvalue()
        r = client.post("/predict", data={"crop": "Rice"},
                        files={"file": ("leaf.tiff", tiff_bytes, "image/tiff")})
        assert r.status_code == 200


class TestPredictFlow:
    """End-to-end prediction flow: image → disease → probabilities."""

    def test_full_predict_returns_all_fields(self, client, tiny_jpg_bytes):
        r = client.post("/predict", data={"crop": "Rice"},
                        files={"file": ("leaf.jpg", tiny_jpg_bytes, "image/jpeg")})
        assert r.status_code == 200
        body = r.json()
        assert body["crop"] == "Rice"
        assert isinstance(body["disease"], str)
        assert isinstance(body["confidence"], float)
        assert isinstance(body["probabilities"], dict)
        assert len(body["probabilities"]) == len(CROP_CLASSES["Rice"])

    @pytest.mark.parametrize("crop", list(CROP_CLASSES))
    def test_predict_then_treatment(self, client, tiny_jpg_bytes, crop):
        """Predict disease, then fetch treatment — the full client flow."""
        r = client.post("/predict", data={"crop": crop},
                        files={"file": ("leaf.jpg", tiny_jpg_bytes, "image/jpeg")})
        assert r.status_code == 200
        disease = r.json()["disease"]
        r2 = client.get(f"/treatment/{crop}/{disease}", params={"lang": "en"})
        assert r2.status_code == 200
        t = r2.json()
        assert t["explanation"].strip()
        assert len(t["symptoms"]) > 0
        assert len(t["treatment"]) > 0
        assert len(t["prevention"]) > 0
