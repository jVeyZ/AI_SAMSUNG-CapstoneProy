"""
CropGuard: AI-Powered Multi-Crop Disease Diagnosis
Samsung Innovation Campus - Capstone Project

End-to-end fine-tuning of ResNet50 for:
  - Tomato (10 classes)
  - Rice (10 classes)
  - Orange (5 classes)

Two phases per crop:
  Phase 1 — head warmup: backbone frozen, train only the MLP head.
  Phase 2 — fine-tuning: unfreeze layer4 (+ head), low backbone LR.

Usage:
    python train.py                  # Train all available crops
    python train.py --crop tomato    # Train only tomato
    python train.py --epochs 40      # Max fine-tuning epochs (phase 2)
    python train.py --ft-lr 1e-4     # Backbone (layer4) learning rate
    python train.py --workers 4      # DataLoader workers (0 to debug)
    python train.py --no-early-stop  # Disable early stopping
"""
import os, argparse, json, copy

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from cropguard.crop_config import CROP_CLASSES, get_crop_data

from cropguard.model_def import (build_model, model_path, TRAIN_TRANSFORM, EVAL_TRANSFORM,
                       MODELS_DIR)

CROP_DATA = get_crop_data()

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = Path.cwd()
DATA_DIR = REPO_ROOT / "src" / "data"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "training_results.json"

RANDOM_STATE = 42

os.makedirs(MODELS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True


def banner(s):
    print(f"\n{'=' * 60}\n  {s}\n{'=' * 60}")


def parse_args():
    p = argparse.ArgumentParser(description="CropGuard Multi-Crop Training")
    p.add_argument("--crop", type=str, default=None,
                   choices=[c.lower() for c in CROP_CLASSES],
                   help="Train only a specific crop (default: all)")
    p.add_argument("--epochs", type=int, default=40,
                   help="Max fine-tuning epochs, phase 2 (default: 40)")
    p.add_argument("--head-epochs", type=int, default=15,
                   help="Max head-warmup epochs, phase 1 (default: 15)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size (default: 32)")
    p.add_argument("--head-lr", type=float, default=1e-3,
                   help="Head learning rate, phase 1 (default: 1e-3)")
    p.add_argument("--ft-lr", type=float, default=1e-4,
                   help="Backbone (layer4) learning rate, phase 2 (default: 1e-4; head uses 10x)")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader worker processes (default: 4; use 0 to debug)")
    p.add_argument("--no-early-stop", action="store_true",
                   help="Disable early stopping")
    p.add_argument("--no-weights", action="store_true",
                   help="Disable class weight balancing")
    return p.parse_args()


# ================================================================
# Figures
# ================================================================
def heatmap(cm, classes, title, save_path, normalize=False):
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
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
            text = f"{cm[i, j]:.2f}" if normalize else str(cm[i, j])
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=7, color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def class_metrics_figure(report, classes, crop_name, save_path):
    """Grouped bar chart of per-class precision/recall/F1 with support counts."""
    prec = [report[c]["precision"] for c in classes]
    rec = [report[c]["recall"] for c in classes]
    f1 = [report[c]["f1-score"] for c in classes]
    support = [int(report[c]["support"]) for c in classes]

    x = np.arange(len(classes))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(9, len(classes) * 1.1), 5))
    ax.bar(x - w, prec, w, label="Precision")
    ax.bar(x, rec, w, label="Recall")
    ax.bar(x + w, f1, w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={s})" for c, s in zip(classes, support)], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-class metrics — {crop_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def confidence_histogram(probs, y_true, y_pred, crop_name, save_path):
    """Max-softmax confidence distribution for correct vs incorrect predictions."""
    conf = probs.max(axis=1)
    correct = y_pred == y_true
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(conf[correct], bins=bins, alpha=0.7, label=f"Correct (n={correct.sum()})", color="#2E8B57")
    ax.hist(conf[~correct], bins=bins, alpha=0.7, label=f"Incorrect (n={(~correct).sum()})", color="#C0392B")
    ax.set_xlabel("Confidence (max softmax)")
    ax.set_ylabel("Count")
    ax.set_title(f"Prediction confidence — {crop_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def misclassified_grid(eval_full, test_indices, y_true, y_pred, classes, crop_name, save_path, max_images=12):
    """Grid of misclassified test images with true -> predicted labels."""
    mis = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p][:max_images]
    if not mis:
        return
    ncols = 4
    nrows = int(np.ceil(len(mis) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, mis):
        img_path = eval_full.samples[test_indices[i]][0]
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"T: {classes[y_true[i]]}\nP: {classes[y_pred[i]]}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(mis):]:
        ax.axis("off")
    fig.suptitle(f"Misclassified test samples — {crop_name} ({int((y_pred != y_true).sum())} total)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def class_distribution_figure(targets, splits, classes, crop_name, save_path):
    """Per-class image counts across train/val/test splits."""
    counts = {name: np.bincount(targets[subset.indices], minlength=len(classes))
              for name, subset in splits.items()}
    x = np.arange(len(classes))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(9, len(classes) * 1.1), 4))
    for k, (name, c) in enumerate(counts.items()):
        ax.bar(x + (k - 1) * w, c, w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Images")
    ax.set_title(f"Class distribution — {crop_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def history_figure(history, phase_boundary, crop_name, save_path):
    """Train/val accuracy and loss; dashed line marks the fine-tuning start."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in ((ax1, "accuracy", "Accuracy"), (ax2, "loss", "Loss")):
        ax.plot(history[key], "b", label="train")
        ax.plot(history[f"val_{key}"], "r", label="val")
        if phase_boundary:
            ax.axvline(phase_boundary - 0.5, color="gray", linestyle="--", alpha=0.7)
            ax.text(phase_boundary - 0.5, ax.get_ylim()[1], " fine-tune", color="gray", fontsize=8, va="top")
        ax.set_title(f"{crop_name} — {title}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


# ================================================================
# Results persistence
# ================================================================
def load_existing_results():
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                return json.load(f).get("models", {})
        except (json.JSONDecodeError, KeyError):
            return {}
    return {}


# ================================================================
# Training loops
# ================================================================
def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over loader. Returns (avg_loss, accuracy). Trains if optimizer given."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_correct += int((logits.argmax(dim=1) == labels).sum())
            total += bs
    return total_loss / total, total_correct / total


def train_phase(model, train_loader, val_loader, criterion, optimizer, max_epochs,
                patience, history, scheduler=None, checkpoint_path=None):
    """Train with early stopping on val_loss; restores best weights. Returns (epochs_ran, best_epoch).

    If checkpoint_path is given, the best weights are also written to disk on
    every improvement, so an interrupted run still leaves a usable model."""
    best_val, best_state, wait = float("inf"), None, 0
    best_epoch, epochs_ran = 0, 0
    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion)
        if scheduler is not None:
            scheduler.step(va_loss)
        for k, v in (("loss", tr_loss), ("accuracy", tr_acc), ("val_loss", va_loss), ("val_accuracy", va_acc)):
            history[k].append(v)
        epochs_ran = epoch
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:>2}/{max_epochs} - loss {tr_loss:.4f} - acc {tr_acc:.4f} "
              f"- val_loss {va_loss:.4f} - val_acc {va_acc:.4f} - lr {lr:.2e}", flush=True)

        if va_loss < best_val - 1e-4:
            best_val, best_epoch, wait = va_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping (best epoch {best_epoch}, val_loss {best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return epochs_ran, best_epoch


# ================================================================
def train_crop(crop_name, args):
    crop_info = CROP_DATA[crop_name]
    num_classes = crop_info["num_classes"]
    crop_data_dir = os.path.join(DATA_DIR, crop_info["data_dir"])

    if not os.path.isdir(crop_data_dir):
        print(f"\n  Data directory not found: {crop_data_dir}. Skipping {crop_name}.")
        print(f"  Run: python setup.py  to download datasets first.")
        return None

    banner(f"TRAINING: {crop_name}  ({num_classes} classes)")

    # ---- Two ImageFolder views of the same root: train view is augmented,
    #      eval view is not. random_split with the same seed produces identical
    #      indices for both, so the split stays consistent. -------------------
    train_full = datasets.ImageFolder(crop_data_dir, transform=TRAIN_TRANSFORM)
    eval_full = datasets.ImageFolder(crop_data_dir, transform=EVAL_TRANSFORM)
    n_total = len(eval_full)
    n_train = int(n_total * 0.75)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val

    split_sizes = [n_train, n_val, n_test]
    train_ds = torch.utils.data.random_split(
        train_full, split_sizes,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )[0]
    val_ds, test_ds = torch.utils.data.random_split(
        eval_full, split_sizes,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )[1:]

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers,
                     pin_memory=(DEVICE.type == "cuda"),
                     persistent_workers=(args.workers > 0))
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    print(f"  Train: {n_train:>6,}  |  Val: {n_val:>5,}  |  Test: {n_test:>5,}")

    # ---- Class weights -----------------------------------------------------
    ytr_raw = np.array(train_full.targets)[train_ds.indices]
    if args.no_weights:
        weight_tensor = None
    else:
        cw = compute_class_weight("balanced", classes=np.unique(ytr_raw), y=ytr_raw)
        weight_tensor = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
        print(f"  Class weights: { {crop_info['classes'][k]: f'{v:.2f}' for k, v in enumerate(cw)} }")

    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)

    # ---- Model --------------------------------------------------------------
    model = build_model(num_classes).to(DEVICE)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    es_head = 5 if not args.no_early_stop else args.head_epochs
    es_ft = 8 if not args.no_early_stop else args.epochs

    # ---- Phase 1: head warmup (backbone frozen) ------------------------------
    print("\n  Phase 1: head warmup (backbone frozen)")
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    opt1 = torch.optim.Adam(model.fc.parameters(), lr=args.head_lr)
    head_epochs, head_best = train_phase(
        model, train_loader, val_loader, criterion, opt1,
        max_epochs=args.head_epochs, patience=es_head, history=history)

    # ---- Phase 2: fine-tune layer4 + head ------------------------------------
    print("\n  Phase 2: fine-tuning layer4 + head")
    for p in model.layer4.parameters():
        p.requires_grad = True
    opt2 = torch.optim.AdamW([
        {"params": model.layer4.parameters(), "lr": args.ft_lr},
        {"params": model.fc.parameters(), "lr": args.ft_lr * 10},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, mode="min", factor=0.5, patience=3)
    ft_epochs, ft_best = train_phase(
        model, train_loader, val_loader, criterion, opt2,
        max_epochs=args.epochs, patience=es_ft, history=history, scheduler=scheduler,
        checkpoint_path=model_path(crop_name))

    # ---- Evaluate -------------------------------------------------------------
    model.eval()
    probs_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(DEVICE))
            probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels_list.append(labels.numpy())
    yte_probs = np.concatenate(probs_list)
    yte_raw = np.concatenate(labels_list)
    yp = yte_probs.argmax(axis=1)

    test_acc = float(np.mean(yp == yte_raw))
    top2_acc = float(top_k_accuracy_score(yte_raw, yte_probs, k=2, labels=list(range(num_classes))))
    report = classification_report(yte_raw, yp, target_names=crop_info["classes"], output_dict=True, zero_division=0)

    print(f"\n  {crop_name} Test Accuracy: {test_acc:.2%}  (top-2: {top2_acc:.2%})")
    print(f"  Phase 1: {head_epochs} epochs (best {head_best})  |  Phase 2: {ft_epochs} epochs (best {ft_best})")
    print(classification_report(yte_raw, yp, target_names=crop_info["classes"], zero_division=0))

    # ---- Save model -----------------------------------------------------------
    torch.save(model.state_dict(), model_path(crop_name))
    print(f"  Saved: {os.path.basename(model_path(crop_name))}")

    # ---- Plots (one directory per crop: results/<crop>/) ----------------------
    crop_results_dir = os.path.join(RESULTS_DIR, crop_name.lower())
    os.makedirs(crop_results_dir, exist_ok=True)

    cm = confusion_matrix(yte_raw, yp)
    heatmap(cm, crop_info["classes"], f"Confusion Matrix — {crop_name}",
            os.path.join(crop_results_dir, "confusion_matrix.png"))
    heatmap(cm, crop_info["classes"], f"Confusion Matrix (normalized) — {crop_name}",
            os.path.join(crop_results_dir, "confusion_matrix_normalized.png"), normalize=True)
    history_figure(history, head_epochs, crop_name,
                   os.path.join(crop_results_dir, "history.png"))
    class_metrics_figure(report, crop_info["classes"], crop_name,
                         os.path.join(crop_results_dir, "class_metrics.png"))
    confidence_histogram(yte_probs, yte_raw, yp, crop_name,
                         os.path.join(crop_results_dir, "confidence_histogram.png"))
    misclassified_grid(eval_full, test_ds.indices, yte_raw, yp, crop_info["classes"], crop_name,
                       os.path.join(crop_results_dir, "misclassified_samples.png"))
    class_distribution_figure(
        np.array(eval_full.targets),
        {"train": train_ds, "val": val_ds, "test": test_ds},
        crop_info["classes"], crop_name,
        os.path.join(crop_results_dir, "class_distribution.png"))

    return {
        "num_classes": num_classes,
        "classes": crop_info["classes"],
        "test_accuracy": test_acc,
        "top2_accuracy": top2_acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {
            c: {
                "precision": report[c]["precision"],
                "recall": report[c]["recall"],
                "f1": report[c]["f1-score"],
                "support": report[c]["support"],
            }
            for c in crop_info["classes"]
        },
        "train_samples": n_train,
        "val_samples": n_val,
        "test_samples": n_test,
        "head_epochs_ran": head_epochs,
        "epochs_ran": ft_epochs,
        "best_epoch": ft_best,
        "finetuned": True,
        "unfrozen": "layer4",
    }


def main():
    args = parse_args()

    banner("CropGuard - Multi-Crop Training Pipeline (fine-tuning)")
    print(f"  Device   : {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    print(f"  HeadLR   : {args.head_lr}  (max {args.head_epochs} epochs)")
    print(f"  FtLR     : {args.ft_lr} (head x10)  (max {args.epochs} epochs)")
    print(f"  Batch    : {args.batch_size}")
    print(f"  Workers  : {args.workers}")
    print(f"  EarlyStop: {not args.no_early_stop}")
    print(f"  Weights  : {not args.no_weights}")

    crops_to_train = [args.crop.capitalize()] if args.crop else list(CROP_DATA.keys())
    new_results = {}

    for crop_name in crops_to_train:
        if crop_name not in CROP_DATA:
            print(f"\n  Unknown crop: {crop_name}. Available: {list(CROP_DATA.keys())}")
            continue

        result = train_crop(crop_name, args)
        if result is None:
            continue
        new_results[crop_name] = result

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


if __name__ == "__main__":
    main()
