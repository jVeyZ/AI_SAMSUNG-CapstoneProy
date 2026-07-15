"""
CropGuard: AI-Powered Multi-Crop Disease Diagnosis
Samsung Innovation Campus - Capstone Project

Trains separate Transfer Learning (ResNet50) models for:
  - Tomato (10 classes)
  - Rice (10 classes)
  - Orange (5 classes)
"""
import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from treatment_db import TREATMENTS

CROP_DATA = {}
for _id, _info in TREATMENTS.items():
    _crop = _info["crop"]
    if _crop not in CROP_DATA:
        CROP_DATA[_crop] = {"classes": [], "treatment_start": _id, "data_dir": _crop.lower()}
    CROP_DATA[_crop]["classes"].append(_info["name"])

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.figsize": (10, 5), "axes.spines.top": False, "axes.spines.right": False})

RANDOM_STATE = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORK_DIR, "data")

def banner(s):
    print(f"\n{'='*60}\n{s}\n{'='*60}")

banner("CropGuard - Multi-Crop Training Pipeline")
print(f"Backend: Keras/{os.environ['KERAS_BACKEND']}")
print(f"CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

resnet = models.resnet50(weights="IMAGENET1K_V2")
if torch.cuda.is_available():
    resnet = resnet.cuda()
resnet.eval()

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

def extract_features(loader):
    feats, lbls_list = [], []
    with torch.no_grad():
        for images, labels in loader:
            if torch.cuda.is_available():
                images = images.cuda()
            f = resnet(images).cpu().numpy()
            feats.append(f)
            lbls_list.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(lbls_list)

results = {}

for crop_name, crop_info in CROP_DATA.items():
    banner(f"TRAINING: {crop_name}")
    num_classes = len(crop_info["classes"])
    crop_data_dir = os.path.join(DATA_DIR, crop_info["data_dir"])

    if not os.path.exists(crop_data_dir):
        print(f"  Data directory not found: {crop_data_dir}. Skipping.")
        continue

    full_ds = datasets.ImageFolder(crop_data_dir, transform=val_transform)
    n_total = len(full_ds)
    n_train = int(n_total * 0.75)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(RANDOM_STATE)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Classes: {num_classes}  |  Train: {n_train}  Val: {n_val}  Test: {n_test}")

    print("  Extracting ResNet50 features...")
    Xtr, ytr_raw = extract_features(train_loader)
    Xv, yv_raw = extract_features(val_loader)
    Xte, yte_raw = extract_features(test_loader)

    ytr = np.zeros((len(ytr_raw), num_classes)); ytr[np.arange(len(ytr_raw)), ytr_raw] = 1
    yv = np.zeros((len(yv_raw), num_classes)); yv[np.arange(len(yv_raw)), yv_raw] = 1
    yte = np.zeros((len(yte_raw), num_classes)); yte[np.arange(len(yte_raw)), yte_raw] = 1

    print(f"  Features: Train {Xtr.shape}  Val {Xv.shape}  Test {Xte.shape}")

    m = Sequential([
        Input(shape=(1000,)),
        Dense(256, activation='relu'), Dropout(0.5),
        Dense(128, activation='relu'), Dropout(0.3),
        Dense(num_classes, activation='softmax'),
    ])
    m.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    m.summary()

    print(f"  Training {crop_name} classifier...")
    h = m.fit(Xtr, ytr, validation_data=(Xv, yv), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

    yp = np.argmax(m.predict(Xte, verbose=0), axis=1)
    test_acc = np.mean(yp == yte_raw)

    print(f"\n  {crop_name} Test Accuracy: {test_acc:.2%}")
    print(classification_report(yte_raw, yp, target_names=crop_info["classes"]))

    model_path = os.path.join(WORK_DIR, "models", f"cropguard_{crop_name.lower()}_classifier.keras")
    m.save(model_path)
    print(f"  Saved: cropguard_{crop_name.lower()}_classifier.keras")

    cm = confusion_matrix(yte_raw, yp)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=crop_info["classes"], yticklabels=crop_info["classes"])
    ax.set_title(f'Confusion Matrix - {crop_name}')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, "results", "figures", f"confusion_matrix_{crop_name.lower()}.png"), dpi=100, bbox_inches='tight')
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(h.history['accuracy'], 'b', label='train'); ax1.plot(h.history['val_accuracy'], 'r', label='val')
    ax1.set_title(f'{crop_name} - Accuracy'); ax1.legend()
    ax2.plot(h.history['loss'], 'b', label='train'); ax2.plot(h.history['val_loss'], 'r', label='val')
    ax2.set_title(f'{crop_name} - Loss'); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, "results", "figures", f"history_{crop_name.lower()}.png"), dpi=100)
    plt.close()

    results[crop_name] = {
        "num_classes": num_classes,
        "classes": crop_info["classes"],
        "test_accuracy": float(test_acc),
        "train_samples": n_train,
        "val_samples": n_val,
        "test_samples": n_test,
    }

torch.save(resnet.state_dict(), os.path.join(WORK_DIR, "models", "cropguard_resnet50.pth"))
print(f"\nSaved: cropguard_resnet50.pth")

results_data = {
    "project": "CropGuard - Multi-Crop Disease Diagnosis",
    "models": results,
}
with open(os.path.join(WORK_DIR, "results", "training_results.json"), "w") as f:
    json.dump(results_data, f, indent=2)
print("Saved: training_results.json")

banner("ALL DONE")
for crop, info in results.items():
    print(f"  {crop}: {info['test_accuracy']:.2%} ({info['num_classes']} classes)")
