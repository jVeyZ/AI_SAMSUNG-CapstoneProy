"""Unit tests for model_def — the shared training/inference contract."""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from model_def import (EVAL_TRANSFORM, TRAIN_TRANSFORM, build_model, model_path)


def test_fc_head_structure():
    model = build_model(7, weights=None)
    assert isinstance(model.fc, nn.Sequential)
    layer_types = [type(l).__name__ for l in model.fc]
    assert layer_types == ["Linear", "ReLU", "Dropout", "Linear", "ReLU", "Dropout", "Linear"]
    assert model.fc[0].in_features == 2048
    assert model.fc[-1].out_features == 7


def test_forward_shape():
    model = build_model(5, weights=None).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 5)


def test_eval_transform_normalizes():
    img = Image.fromarray((np.random.rand(300, 200, 3) * 255).astype("uint8"))
    t = EVAL_TRANSFORM(img)
    assert t.shape == (3, 224, 224)
    # ImageNet normalization pushes values outside [0, 1]
    assert t.min() < 0 or t.max() > 1


def test_train_transform_shape():
    img = Image.fromarray((np.random.rand(300, 200, 3) * 255).astype("uint8"))
    t = TRAIN_TRANSFORM(img)
    assert t.shape == (3, 224, 224)


def test_model_path():
    p = model_path("Rice")
    assert p.endswith("cropguard_rice_model.pth")
