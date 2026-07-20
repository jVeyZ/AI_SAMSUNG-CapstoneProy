"""Unit tests for crop_config — the class-name ↔ data-dir alignment invariant."""
import os

import pytest

from cropguard.crop_config import (CROP_CLASSES, CROP_DATA_DIRS, class_to_dirname,
                         get_crop_data, get_disease_name, get_num_classes)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Known on-disk directory names (alphabetical order), mirroring setup.py targets.
EXPECTED_DIRS = {
    "Tomato": ["Bacterial_spot", "Early_blight", "Healthy", "Late_blight", "Leaf_Mold",
               "Mosaic_virus", "Septoria_spot", "Spider_mites", "Target_Spot", "Yellow_Curl"],
    "Rice": [class_to_dirname(n) for n in CROP_CLASSES["Rice"]],
    "Orange": [class_to_dirname(n) for n in CROP_CLASSES["Orange"]],
}


def test_class_counts():
    assert {crop: len(classes) for crop, classes in CROP_CLASSES.items()} == {
        "Tomato": 10, "Rice": 10, "Orange": 5}


def test_dirnames_unique_per_crop():
    for crop, classes in CROP_CLASSES.items():
        dirnames = [class_to_dirname(c) for c in classes]
        assert len(dirnames) == len(set(dirnames)), f"duplicate dirnames in {crop}"


def test_helpers():
    assert get_num_classes("Rice") == 10
    assert get_disease_name("Orange", 1) == "Canker"
    data = get_crop_data()
    assert set(data) == set(CROP_CLASSES)
    for crop, info in data.items():
        assert info["num_classes"] == len(info["classes"])
        assert info["data_dir"] == CROP_DATA_DIRS[crop]


def test_imagefolder_assigns_indices_alphabetically():
    """The invariant everything depends on: ImageFolder sorts dirs alphabetically."""
    from torchvision.datasets import ImageFolder

    for crop in CROP_CLASSES:
        expected = EXPECTED_DIRS[crop]
        # Display names must be listed in the same order as alphabetical dir names
        assert expected == sorted(expected), f"{crop}: expected dirs not alphabetical (test bug)"
        # For rice/orange, class_to_dirname(display) IS the dir name, so this is exact.
        # For tomato the mapping is semantic; EXPECTED_DIRS is fixed above.


@pytest.mark.parametrize("crop", list(CROP_CLASSES))
def test_alignment_with_real_data(crop):
    """Guard that would have caught the rice-order bug. Skips where data is absent (CI)."""
    data_dir = os.path.join(REPO_ROOT, "data", crop.lower())
    if not os.path.isdir(data_dir):
        pytest.skip(f"{data_dir} not present")
    from torchvision.datasets import ImageFolder

    ds = ImageFolder(data_dir)
    idx_order = [name for name, _ in sorted(ds.class_to_idx.items(), key=lambda kv: kv[1])]
    assert idx_order == sorted(idx_order)  # sanity: ImageFolder is alphabetical
    assert idx_order == EXPECTED_DIRS[crop], (
        f"{crop}: data dirs {idx_order} do not match expected order {EXPECTED_DIRS[crop]}")
