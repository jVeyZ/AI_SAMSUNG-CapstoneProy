"""
CropGuard: AI-Powered Multi-Crop Disease Diagnosis
Samsung Innovation Campus - Capstone Project

Trains separate Transfer Learning (ResNet50) models for:
  - Tomato (10 classes)
  - Rice (10 classes)
  - Orange (5 classes)

Usage:
    python train.py                  # Train all available crops
    python train.py --crop tomato    # Train only tomato
    python train.py --crop rice      # Train only rice
    python train.py --epochs 20      # Custom epochs
    python train.py --no-early-stop  # Disable early stopping
"""
import os, argparse, json
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from crop_config import CROP_CLASSES, CROP_DATA_DIRS, get_crop_data

CROP_DATA = get_crop_data()

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORK_DIR, "data")
MODELS_DIR = os.path.join(WORK_DIR, "models")
RESULTS_DIR = os.path.join(WORK_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
RESULTS_FILE = os.path.join(RESULTS_DIR, "training_results.json")
RESNET_PATH = os.path.join(MODELS_DIR, "cropguard_resnet50.pth")

RANDOM_STATE = 42
IMG_SIZE = (224, 224)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def banner(s):
    print(f"\n{'=' * 60}\n  {s}\n{'=' * 60}")


def parse_args():
    p = argparse.ArgumentParser(description="CropGuard Multi-Crop Training")
    p.add_argument("--crop", type=str, default=None,
                   choices=[c.lower() for c in CROP_CLASSES],
                   help="Train only a specific crop (default: all)")
    p.add_argument("--epochs", type=int, default=100,
                   help="Maximum training epochs (default: 100)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size (default: 32)")
    p.add_argument("--no-early-stop", action="store_true",
                   help="Disable early stopping")
    p.add_argument("--no-weights", action="store_true",
                   help="Disable class weight balancing")
    return p.parse_args()


def heatmap(cm, classes, title, save_path):
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.9), max(6, len(classes) * 0.7)))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=7, color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def load_existing_results():
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                return json.load(f).get("models", {})
        except (json.JSONDecodeError, KeyError):
            return {}
    return {}


# ================================================================
args = parse_args()

banner("CropGuard - Multi-Crop Training Pipeline")
print(f"  Backend  : Keras/{os.environ['KERAS_BACKEND']}")
print(f"  CUDA     : {torch.cuda.is_available()}")
print(f"  Epochs   : {args.epochs}")
print(f"  Batch    : {args.batch_size}")
print(f"  EarlyStop: {not args.no_early_stop}")
print(f"  Weights  : {not args.no_weights}")

# ---- Feature extractor (shared across all crops) ------------------------
resnet = models.resnet50(weights="IMAGENET1K_V2")
if torch.cuda.is_available():
    resnet = resnet.cuda()
resnet.eval()

# Save ResNet state dict early — it never changes during training
torch.save(resnet.state_dict(), RESNET_PATH)
print(f"\n  Saved ResNet50 backbone: cropguard_resnet50.pth")

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])


def extract_features(loader):
    feats, lbls = [], []
    with torch.no_grad():
        for images, labels in loader:
            if torch.cuda.is_available():
                images = images.cuda()
            f = resnet(images).cpu().numpy()
            feats.append(f)
            lbls.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(lbls)


# ---- Determine crops to train -------------------------------------------
existing_results = load_existing_results()
crops_to_train = [args.crop.capitalize()] if args.crop else list(CROP_DATA.keys())
new_results = {}

for crop_name in crops_to_train:
    if crop_name not in CROP_DATA:
        print(f"\n  Unknown crop: {crop_name}. Available: {list(CROP_DATA.keys())}")
        continue

    crop_info = CROP_DATA[crop_name]
    num_classes = crop_info["num_classes"]
    crop_data_dir = os.path.join(DATA_DIR, crop_info["data_dir"])

    if not os.path.isdir(crop_data_dir):
        print(f"\n  Data directory not found: {crop_data_dir}. Skipping {crop_name}.")
        print(f"  Run: python setup.py  to download datasets first.")
        continue

    banner(f"TRAINING: {crop_name}  ({num_classes} classes)")

    # ---- Load with eval transform to build dataset, then apply aug to train subsets ----
    full_ds = datasets.ImageFolder(crop_data_dir, transform=eval_transform)
    n_total = len(full_ds)
    n_train = int(n_total * 0.75)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )

    # Apply augmentation only to training subset
    train_ds.dataset.transform = train_transform

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"  Train: {n_train:>6,}  |  Val: {n_val:>5,}  |  Test: {n_test:>5,}")

    # ---- Feature extraction ---------------------------------------------
    print("  Extracting ResNet50 features...")
    Xtr, ytr_raw = extract_features(train_loader)
    Xv, yv_raw = extract_features(val_loader)
    Xte, yte_raw = extract_features(test_loader)

    # One-hot encode
    ytr = np.eye(num_classes)[ytr_raw]
    yv = np.eye(num_classes)[yv_raw]
    yte = np.eye(num_classes)[yte_raw]

    print(f"  Feature shapes — Train: {Xtr.shape}  Val: {Xv.shape}  Test: {Xte.shape}")

    # ---- Class weights --------------------------------------------------
    class_weight = None
    if not args.no_weights:
        class_weight = compute_class_weight(
            "balanced", classes=np.unique(ytr_raw), y=ytr_raw
        )
        class_weight = {i: w for i, w in enumerate(class_weight)}
        print(f"  Class weights: { {crop_info['classes'][k]: f'{v:.2f}' for k, v in class_weight.items()} }")

    # ---- Build classifier -----------------------------------------------
    m = Sequential([
        Input(shape=(1000,)),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    m.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    m.summary()

    # ---- Callbacks ------------------------------------------------------
    callbacks = []
    if not args.no_early_stop:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ))
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1,
    ))

    # ---- Train ----------------------------------------------------------
    print(f"  Training {crop_name} classifier...")
    h = m.fit(
        Xtr, ytr,
        validation_data=(Xv, yv),
        batch_size=args.batch_size,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # ---- Evaluate -------------------------------------------------------
    yp = m.predict(Xte, verbose=0).argmax(axis=1)
    test_acc = float(np.mean(yp == yte_raw))
    n_epochs_ran = len(h.history["loss"])
    best_epoch = int(np.argmin(h.history["val_loss"])) + 1

    print(f"\n  {crop_name} Test Accuracy: {test_acc:.2%}  (trained {n_epochs_ran} epochs, best epoch {best_epoch})")
    print(classification_report(yte_raw, yp, target_names=crop_info["classes"]))

    # ---- Save model -----------------------------------------------------
    model_path = os.path.join(MODELS_DIR, f"cropguard_{crop_name.lower()}_classifier.keras")
    m.save(model_path)
    print(f"  Saved: cropguard_{crop_name.lower()}_classifier.keras")

    # ---- Plots ----------------------------------------------------------
    cm = confusion_matrix(yte_raw, yp)
    heatmap(cm, crop_info["classes"], f"Confusion Matrix — {crop_name}",
            os.path.join(FIGURES_DIR, f"confusion_matrix_{crop_name.lower()}.png"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(h.history["accuracy"], "b", label="train")
    ax1.plot(h.history["val_accuracy"], "r", label="val")
    ax1.set_title(f"{crop_name} — Accuracy")
    ax1.legend()
    ax2.plot(h.history["loss"], "b", label="train")
    ax2.plot(h.history["val_loss"], "r", label="val")
    ax2.set_title(f"{crop_name} — Loss")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"history_{crop_name.lower()}.png"), dpi=100)
    plt.close()

    # ---- Store results and save incrementally --------------------------
    new_results[crop_name] = {
        "num_classes": num_classes,
        "classes": crop_info["classes"],
        "test_accuracy": test_acc,
        "train_samples": n_train,
        "val_samples": n_val,
        "test_samples": n_test,
    }

    # Save after each crop so a crash mid-pipeline doesn't lose earlier results
    merged = {**load_existing_results(), **new_results}
    results_data = {"project": "CropGuard - Multi-Crop Disease Diagnosis", "models": merged}
    with open(RESULTS_FILE, "w") as f:
        json.dump(results_data, f, indent=2)

# ---- Final summary ---------------------------------------------------------
merged = load_existing_results()
banner("TRAINING COMPLETE")
for crop, info in merged.items():
    print(f"  {crop:10s}: {info['test_accuracy']:.2%}  ({info['num_classes']} classes, {info['test_samples']} test)")
