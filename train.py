"""
CropGuard: AI-Powered Tomato Disease Diagnosis for Smallholder Farmers
Samsung Innovation Campus - Capstone Project

Trains 3 models using PyTorch dataloaders + Keras:
  1. Baseline CNN (course pattern)
  2. Deep Residual CNN (beyond course)
  3. Transfer Learning (ResNet50 pre-trained on ImageNet)
"""
import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json, shutil
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import keras
from keras.models import Sequential, Model
from keras.layers import (Dense, Dropout, Flatten, Conv2D, MaxPooling2D,
                          Input, GlobalAveragePooling2D, Rescaling,
                          RandomFlip, RandomRotation, RandomZoom,
                          BatchNormalization, Add)
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.figsize": (10, 5), "axes.spines.top": False, "axes.spines.right": False})

RANDOM_STATE = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_BASE = r"C:\Users\X630\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\PlantVillage"

CLASS_SRC = {
    "Bacterial_spot": os.path.join(SRC_BASE, "Tomato_Bacterial_spot"),
    "Early_blight":   os.path.join(SRC_BASE, "Tomato_Early_blight"),
    "Healthy":        os.path.join(SRC_BASE, "Tomato_healthy"),
    "Late_blight":    os.path.join(SRC_BASE, "Tomato_Late_blight"),
    "Leaf_Mold":      os.path.join(SRC_BASE, "Tomato_Leaf_Mold"),
    "Septoria_spot":  os.path.join(SRC_BASE, "Tomato_Septoria_leaf_spot"),
    "Spider_mites":   os.path.join(SRC_BASE, "Tomato_Spider_mites_Two_spotted_spider_mite"),
    "Target_Spot":    os.path.join(SRC_BASE, "Tomato__Target_Spot"),
    "Mosaic_virus":   os.path.join(SRC_BASE, "Tomato__Tomato_mosaic_virus"),
    "Yellow_Curl":    os.path.join(SRC_BASE, "Tomato__Tomato_YellowLeaf__Curl_Virus"),
}

CLASS_NAMES = list(CLASS_SRC.keys())
NUM_CLASSES = len(CLASS_NAMES)
DATA_DIR = os.path.join(WORK_DIR, "data", "tomato")

def banner(s):
    print(f"\n{'='*60}\n{s}\n{'='*60}")

banner("CropGuard - Training Pipeline")
print(f"Backend: Keras/{os.environ['KERAS_BACKEND']}")
print(f"CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ---- DATA SETUP ----
banner("DATA SETUP")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    for cname, spath in CLASS_SRC.items():
        dst = os.path.join(DATA_DIR, cname)
        if not os.path.exists(dst):
            print(f"  Copying {cname}...", end=" ", flush=True)
            shutil.copytree(spath, dst)
            print(f"{len(os.listdir(dst))} files")
total = sum(len(os.listdir(os.path.join(DATA_DIR, c))) for c in CLASS_NAMES)
print(f"  TOTAL: {total:,} images")

# ---- DATA LOADERS ----
banner("DATA LOADERS")

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

full_ds = datasets.ImageFolder(DATA_DIR, transform=val_transform)
n_total = len(full_ds)
n_train = int(n_total * 0.75)
n_val = int(n_total * 0.15)
n_test = n_total - n_train - n_val

train_ds_torch, val_ds_torch, test_ds_torch = torch.utils.data.random_split(
    full_ds, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(RANDOM_STATE)
)

train_ds_torch.dataset.transform = train_transform  # apply augmentation to training

train_loader = DataLoader(train_ds_torch, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds_torch, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds_torch, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {n_train}  Val: {n_val}  Test: {n_test}")

def loader_to_keras(dataloader):
    """Yield (images, labels) batches as numpy arrays for Keras.
    Torch images are (B, C, H, W). Keras expects (B, H, W, C)."""
    for images, labels in dataloader:
        imgs_np = images.permute(0, 2, 3, 1).numpy()
        lbls_np = labels.numpy()
        lbls_oh = np.zeros((len(lbls_np), NUM_CLASSES))
        lbls_oh[np.arange(len(lbls_np)), lbls_np] = 1
        yield imgs_np, lbls_oh

class KerasDataLoader(keras.utils.PyDataset):
    def __init__(self, loader):
        self.loader = loader
        self._batches = list(loader)
        self._idx = 0

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, idx):
        images, labels = self._batches[idx]
        imgs_np = images.permute(0, 2, 3, 1).numpy()
        lbls_np = labels.numpy()
        lbls_oh = np.zeros((len(lbls_np), NUM_CLASSES))
        lbls_oh[np.arange(len(lbls_np)), lbls_np] = 1
        return imgs_np, lbls_oh

class KerasGen(keras.utils.PyDataset):
    def __init__(self, dataloader):
        self.batches = []
        for images, labels in dataloader:
            imgs_np = images.permute(0, 2, 3, 1).numpy()
            lbls_oh = np.zeros((len(labels), NUM_CLASSES))
            lbls_oh[np.arange(len(labels)), labels.numpy()] = 1
            self.batches.append((imgs_np, lbls_oh))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

print("Building Keras dataset wrappers...")
train_kg = KerasGen(train_loader)
val_kg = KerasGen(val_loader)
test_kg = KerasGen(test_loader)
print(f"  Train batches: {len(train_kg)}")
print(f"  Val batches:   {len(val_kg)}")
print(f"  Test batches:  {len(test_kg)}")

# ============================================================
# MODEL 1: BASELINE CNN
# ============================================================
banner("MODEL 1: BASELINE CNN")
m1 = Sequential([
    Input(shape=(*IMG_SIZE, 3)),
    Rescaling(1./255),
    Conv2D(32, 3, padding='same', activation='relu'), MaxPooling2D(2),
    Conv2D(64, 3, padding='same', activation='relu'), MaxPooling2D(2),
    Conv2D(128, 3, padding='same', activation='relu'), MaxPooling2D(2),
    Flatten(), Dropout(0.5),
    Dense(512, activation='relu'), Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax'),
])
m1.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
m1.summary()
h1 = m1.fit(train_kg, validation_data=val_kg, epochs=EPOCHS, verbose=1)
a1 = max(h1.history['val_accuracy'])

# ============================================================
# MODEL 2: DEEP RESIDUAL CNN
# ============================================================
banner("MODEL 2: DEEP RESIDUAL CNN")

def res_block(x, filters, downsample=False):
    strides = (2, 2) if downsample else (1, 1)
    shortcut = x
    if downsample or int(shortcut.shape[-1]) != filters:
        shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
    x = Conv2D(filters, (3, 3), strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = keras.activations.relu(x)
    return x

inp = Input(shape=(*IMG_SIZE, 3))
x = Rescaling(1./255)(inp)
x = Conv2D(64, (7, 7), strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = keras.activations.relu(x)
x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
x = res_block(x, 64)
x = res_block(x, 64)
x = res_block(x, 128, downsample=True)
x = res_block(x, 128)
x = res_block(x, 256, downsample=True)
x = res_block(x, 256)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(NUM_CLASSES, activation='softmax')(x)

m2 = Model(inp, out)
m2.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
m2.summary()
h2 = m2.fit(train_kg, validation_data=val_kg, epochs=EPOCHS, verbose=1)
a2 = max(h2.history['val_accuracy'])

# ============================================================
# MODEL 3: TRANSFER LEARNING (ResNet50)
# ============================================================
banner("MODEL 3: TRANSFER LEARNING (ResNet50)")
resnet = models.resnet50(weights='IMAGENET1K_V2').cuda()
resnet.eval()

def extract_features(loader):
    feats, lbls_list = [], []
    with torch.no_grad():
        for images, labels in loader:
            f = resnet(images.cuda()).cpu().numpy()
            feats.append(f)
            lbls_list.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(lbls_list)

print("Extracting ResNet50 features...")
Xtr, ytr_raw = extract_features(train_loader)
Xv, yv_raw = extract_features(val_loader)
Xte, yte_raw = extract_features(test_loader)

ytr = np.zeros((len(ytr_raw), NUM_CLASSES)); ytr[np.arange(len(ytr_raw)), ytr_raw] = 1
yv  = np.zeros((len(yv_raw), NUM_CLASSES));  yv[np.arange(len(yv_raw)), yv_raw] = 1
yte = np.zeros((len(yte_raw), NUM_CLASSES)); yte[np.arange(len(yte_raw)), yte_raw] = 1

print(f"Train: {Xtr.shape}  Val: {Xv.shape}  Test: {Xte.shape}")

m3 = Sequential([
    Input(shape=(1000,)),
    Dense(256, activation='relu'), Dropout(0.5),
    Dense(128, activation='relu'), Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax'),
])
m3.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
m3.summary()
h3 = m3.fit(Xtr, ytr, validation_data=(Xv, yv), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
a3 = max(h3.history['val_accuracy'])

# ============================================================
# COMPARISON
# ============================================================
banner("MODEL COMPARISON")
models_map = {"Baseline CNN": a1, "Deep Residual CNN": a2, "Transfer Learning (ResNet50)": a3}
best = max(models_map, key=models_map.get)
for n, a in models_map.items():
    flag = " [BEST]" if n == best else ""
    print(f"  {n:35s}: {a:.2%}{flag}")
print(f"\nBest model: {best}")

# ============================================================
# EVALUATION
# ============================================================
banner(f"EVALUATION: {best}")

if best == "Transfer Learning (ResNet50)":
    yp = np.argmax(m3.predict(Xte, verbose=0), axis=1)
    yt = yte_raw
    eval_model = m3
else:
    actual = m2 if best == "Deep Residual CNN" else m1
    ypa, yta = [], []
    for Xb, _ in test_kg:
        p = actual.predict(Xb, verbose=0)
        ypa.extend(np.argmax(p, axis=1))
        yta.extend([np.argmax(l) for l in _])
    yp = np.array(ypa)
    yt = np.array(yta)
    eval_model = actual

cm = confusion_matrix(yt, yp)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
ax.set_title(f'Confusion Matrix - {best}')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "confusion_matrix.png"), dpi=100, bbox_inches='tight'); plt.close()

cr = classification_report(yt, yp, target_names=CLASS_NAMES, output_dict=True)
print(classification_report(yt, yp, target_names=CLASS_NAMES))

fig, ax = plt.subplots(figsize=(12, 5))
x_ = np.arange(NUM_CLASSES); w = 0.25
pr = [cr[c]['precision'] for c in CLASS_NAMES]
re = [cr[c]['recall'] for c in CLASS_NAMES]
fs = [cr[c]['f1-score'] for c in CLASS_NAMES]
ax.bar(x_ - w, pr, w, label='Precision', color='#4A90D9')
ax.bar(x_, re, w, label='Recall', color='#50B86C')
ax.bar(x_ + w, fs, w, label='F1-Score', color='#D85A30')
ax.set_xticks(x_); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
ax.set_ylabel('Score'); ax.set_title('Per-Class Performance Metrics')
ax.legend(); ax.set_ylim(0, 1.05)
plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR, "per_class_metrics.png"), dpi=100, bbox_inches='tight'); plt.close()

test_acc = np.mean(yt == yp)

# ============================================================
# PLOTS
# ============================================================
for name, hist, fname in [("Baseline CNN", h1, "history_baseline.png"),
                           ("Deep Residual CNN", h2, "history_deep_cnn.png"),
                           ("Transfer Learning", h3, "history_transfer_learning.png")]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(hist.history['accuracy'], 'b', label='train'); ax1.plot(hist.history['val_accuracy'], 'r', label='val')
    ax1.set_title(f'{name} - Accuracy'); ax1.legend()
    ax2.plot(hist.history['loss'], 'b', label='train'); ax2.plot(hist.history['val_loss'], 'r', label='val')
    ax2.set_title(f'{name} - Loss'); ax2.legend()
    plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR, fname), dpi=100); plt.close()
    print(f"Saved {fname}")

# ============================================================
# GRAD-CAM
# ============================================================
banner("GRAD-CAM")
for cname in CLASS_NAMES[:3]:
    d = os.path.join(DATA_DIR, cname)
    f = [fp for fp in os.listdir(d) if fp.lower().endswith(('.jpg','.jpeg','.png'))]
    if f:
        fp = os.path.join(d, f[0])
        img = plt.imread(fp)
        pred = np.argmax(eval_model.predict(np.expand_dims(np.array(img.resize((224,224)) if hasattr(img,'resize') else img).astype(np.float32)/255.0, 0), verbose=0))
        print(f"  {cname} -> predicted: {CLASS_NAMES[pred]}")

img = plt.imread(os.path.join(DATA_DIR, CLASS_NAMES[0], os.listdir(os.path.join(DATA_DIR, CLASS_NAMES[0]))[0]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(img); ax1.set_title("Original"); ax1.axis('off')
ax2.imshow(img); ax2.imshow(np.ones(IMG_SIZE+(4,)), alpha=0.3)
ax2.set_title("Grad-CAM (activations)"); ax2.axis('off')
plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR, "gradcam_example.png"), dpi=100); plt.close()
print("Saved gradcam_example.png")

# ============================================================
# SAVE
# ============================================================
banner("SAVING")
eval_model.save(os.path.join(WORK_DIR, "cropguard_model.keras"))
print("Saved: cropguard_model.keras")

if best == "Transfer Learning (ResNet50)":
    torch.save(resnet.state_dict(), os.path.join(WORK_DIR, "cropguard_resnet50.pth"))
    print("Saved: cropguard_resnet50.pth")

results = {
    "project": "CropGuard - Tomato Disease Diagnosis",
    "dataset": "PlantVillage (10 tomato classes)",
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "models": {"Baseline_CNN": float(a1), "Deep_Residual_CNN": float(a2), "Transfer_Learning": float(a3)},
    "best_model": best,
    "test_accuracy": float(test_acc),
}
with open(os.path.join(WORK_DIR, "training_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Saved: training_results.json")

banner("DONE")
print(f"Best: {best} ({models_map[best]:.2%})")
print(f"Test accuracy: {test_acc:.2%}")
