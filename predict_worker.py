import os, sys, json, traceback
os.environ["KERAS_BACKEND"] = "torch"

image_path = sys.argv[1]

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(WORK_DIR, "cropguard_model.keras")
RESNET_PATH = os.path.join(WORK_DIR, "cropguard_resnet50.pth")

from treatment_db import TREATMENTS

try:
    import torch
    from torchvision import models, transforms
    from PIL import Image
    from keras.models import load_model

    device = torch.device("cpu")
    resnet = models.resnet50(weights="IMAGENET1K_V2")
    if os.path.exists(RESNET_PATH):
        resnet.load_state_dict(torch.load(RESNET_PATH, map_location=device))
    resnet = resnet.to(device).eval()

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

    result = {
        "ok": True,
        "pred_class": pred_class,
        "conf": conf,
        "preds": preds.tolist(),
        "name": TREATMENTS[pred_class]["name"],
        "symptoms": TREATMENTS[pred_class]["symptoms"],
        "treatment": TREATMENTS[pred_class]["treatment"],
        "prevention": TREATMENTS[pred_class]["prevention"],
    }
    print(json.dumps(result))
except Exception as e:
    err = {"ok": False, "error": str(e), "trace": traceback.format_exc()}
    print(json.dumps(err))
    sys.exit(1)
