"""
Standalone prediction worker (for subprocess-based inference or debugging).

Usage: python predict_worker.py <image_path> <crop>
       python predict_worker.py path/to/leaf.jpg Tomato
"""
import os, sys, json, traceback

image_path = sys.argv[1]
crop = sys.argv[2] if len(sys.argv) > 2 else "Tomato"
from cropguard.crop_config import get_disease_name, get_num_classes

from cropguard.model_def import build_model, model_path, EVAL_TRANSFORM

try:
    import torch
    from PIL import Image

    device = torch.device("cpu")

    if not os.path.exists(model_path(crop)):
        raise FileNotFoundError(f"Model not found: {model_path(crop)}")

    model = build_model(get_num_classes(crop), weights=None)
    model.load_state_dict(torch.load(model_path(crop), map_location=device, weights_only=True))
    model = model.to(device).eval()

    img = Image.open(image_path)
    img_tensor = EVAL_TRANSFORM(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = torch.softmax(model(img_tensor), dim=1)[0].numpy()

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
