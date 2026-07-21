import os, sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


@pytest.fixture(scope="session")
def random_models_dir(tmp_path_factory):
    """Random-weight models for all crops in a temp dir.

    Lets server tests run without the real ~92 MB fine-tuned models
    (which are gitignored and unavailable in CI).
    """
    import torch
    from cropguard.crop_config import CROP_CLASSES, get_num_classes
    from cropguard.model_def import build_model

    d = tmp_path_factory.mktemp("models")
    for crop in CROP_CLASSES:
        model = build_model(get_num_classes(crop), weights=None)
        torch.save(model.state_dict(), d / f"cropguard_{crop.lower()}_model.pth")
    return str(d)


@pytest.fixture(scope="session")
def tiny_image_bytes():
    """A small valid PNG generated at runtime (nothing binary committed)."""
    import io
    import numpy as np
    from PIL import Image

    arr = (np.random.RandomState(0).rand(96, 96, 3) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def tiny_jpg_bytes():
    """A small valid JPEG generated at runtime."""
    import io
    import numpy as np
    from PIL import Image

    arr = (np.random.RandomState(0).rand(96, 96, 3) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()
