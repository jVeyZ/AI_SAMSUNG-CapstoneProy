"""
CropGuard — shared model and transform definitions (torch-only).

Used by train.py, app.py, and predict_worker.py so the model architecture
and input preprocessing stay identical between training and inference.
Change the head architecture ONLY here.
"""
import os

import torch
import torch.nn as nn
from torchvision import models, transforms

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("CROPGUARD_MODELS_DIR", os.path.join(REPO_ROOT, "models"))

IMG_SIZE = (224, 224)
# Pretrained ResNet50 expects ImageNet-normalized inputs — do not remove.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_model(num_classes, weights="IMAGENET1K_V2"):
    """ResNet50 with the MLP classifier head in place of the ImageNet fc."""
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )
    return model


def model_path(crop_name):
    return os.path.join(MODELS_DIR, f"cropguard_{crop_name.lower()}_model.pth")
