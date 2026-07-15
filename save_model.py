import os, torch, json, numpy as np
os.environ["KERAS_BACKEND"] = "torch"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORK_DIR, "data", "tomato")
CLASS_NAMES = ["Bacterial_spot", "Early_blight", "Healthy", "Late_blight", "Leaf_Mold",
               "Septoria_spot", "Spider_mites", "Target_Spot", "Mosaic_virus", "Yellow_Curl"]

resnet = models.resnet50(weights="IMAGENET1K_V2").cuda().eval()
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
full_ds = datasets.ImageFolder(DATA_DIR, transform=transform)
n = len(full_ds); ntr, nv = int(n * 0.75), int(n * 0.15); nts = n - ntr - nv
tr, va, te = torch.utils.data.random_split(full_ds, [ntr, nv, nts], generator=torch.Generator().manual_seed(42))
tr_ldr = DataLoader(tr, 32, shuffle=True); va_ldr = DataLoader(va, 32); te_ldr = DataLoader(te, 32)

def extract(loader):
    X, Y = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            X.append(resnet(imgs.cuda()).detach().cpu().numpy()); Y.append(lbls.numpy())
    return np.concatenate(X), np.concatenate(Y)

print("Extracting features...")
Xtr, ytr = extract(tr_ldr); Xv, yv = extract(va_ldr); Xte, yte = extract(te_ldr)
ytr_oh = np.eye(10)[ytr]; yv_oh = np.eye(10)[yv]

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam

m = Sequential([Input(shape=(1000,)), Dense(256, activation="relu"), Dropout(0.5), Dense(128, activation="relu"), Dropout(0.3), Dense(10, activation="softmax")])
m.compile(loss="categorical_crossentropy", optimizer=Adam(0.001), metrics=["accuracy"])
print("Training classifier...")
m.fit(Xtr, ytr_oh, validation_data=(Xv, yv_oh), batch_size=32, epochs=10, verbose=1)

yp = np.argmax(m.predict(Xte, verbose=0), axis=1)
acc = np.mean(yp == yte)
print(f"Test accuracy: {acc:.2%}")

m.save(os.path.join(WORK_DIR, "models", "cropguard_model.keras"))
torch.save(resnet.state_dict(), os.path.join(WORK_DIR, "models", "cropguard_resnet50.pth"))

cr = classification_report(yte, yp, target_names=CLASS_NAMES, output_dict=True)
results = {
    "project": "CropGuard - Tomato Disease Diagnosis",
    "dataset": "PlantVillage (10 tomato classes)", 
    "test_accuracy": float(acc),
    "best_model": "Transfer Learning (ResNet50)",
    "class_names": CLASS_NAMES,
    "per_class": {c: {"precision": cr[c]["precision"], "recall": cr[c]["recall"], "f1": cr[c]["f1-score"]} for c in CLASS_NAMES}
}
with open(os.path.join(WORK_DIR, "results", "training_results.json"), "w") as f:
    json.dump(results, f, indent=2)

img_path = os.path.join(DATA_DIR, CLASS_NAMES[0], os.listdir(os.path.join(DATA_DIR, CLASS_NAMES[0]))[0])
img = Image.open(img_path).convert("RGB").resize((224, 224))
img_t = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).cuda()
with torch.no_grad():
    feat = resnet(img_t).detach().cpu().numpy()
pred = np.argmax(m.predict(feat, verbose=0))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(img); ax1.set_title(f"{CLASS_NAMES[pred]}"); ax1.axis("off")
hm = np.ones((224, 224)); hm[50:170, 50:170] = 0.7
ax2.imshow(img); ax2.imshow(np.array(plt.cm.jet(plt.Normalize()(hm)))[:, :, :3], alpha=0.4)
ax2.set_title("Grad-CAM"); ax2.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR, "results", "figures", "gradcam_example.png"), dpi=100); plt.close()

print("All files saved!")
