"""
Standalone prediction worker (for subprocess-based inference or debugging).

Usage: python predict_worker.py <image_path> <crop>
       python predict_worker.py path/to/leaf.jpg Tomato
"""
import os, sys, json, traceback
os.environ["KERAS_BACKEND"] = "torch"

image_path = sys.argv[1]
crop = sys.argv[2] if len(sys.argv) > 2 else "Tomato"

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(WORK_DIR, "models", f"cropguard_{crop.lower()}_classifier.keras")
RESNET_PATH = os.path.join(WORK_DIR, "models", "cropguard_resnet50.pth")

from crop_config import get_disease_name, CROP_CLASSES

try:
    import torch
    from torchvision import models, transforms
    from PIL import Image
    from keras.models import load_model

    device = torch.device("cpu")
    resnet = models.resnet50(weights=None)
    if os.path.exists(RESNET_PATH):
        resnet.load_state_dict(torch.load(RESNET_PATH, map_location=device, weights_only=True))
    else:
        resnet = models.resnet50(weights="IMAGENET1K_V2")
    resnet = resnet.to(device).eval()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    classifier = load_model(MODEL_PATH)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path)
    img_tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(img_tensor).cpu().numpy()
    preds = classifier.predict(features, verbose=0)[0]
    pred_class = int(preds.argmax())
    conf = float(preds[pred_class])
    disease_name = get_disease_name(crop, pred_class)

    result = {
        "ok": True,
        "crop": crop,
        "pred_class": pred_class,
        "conf": conf,
        "preds": preds.tolist(),
        "disease": disease_name,
    }
    print(json.dumps(result, ensure_ascii=False))
except Exception as e:
    err = {"ok": False, "error": str(e), "trace": traceback.format_exc()}
    print(json.dumps(err, ensure_ascii=False))
    sys.exit(1)
