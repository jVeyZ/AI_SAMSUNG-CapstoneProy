"""
CropGuard — Crop class configuration.
Maps crops to their disease classes and data directories.
No treatment data — the LLM handles symptoms, treatment, and prevention dynamically.
"""

# Disease class names per crop, in ImageFolder alphabetical order.
# These MUST match the subdirectory names in data/<crop>/.
CROP_CLASSES = {
    "Tomato": [
        "Bacterial Spot",
        "Early Blight",
        "Healthy",
        "Late Blight",
        "Leaf Mold",
        "Mosaic Virus",
        "Septoria Leaf Spot",
        "Spider Mites",
        "Target Spot",
        "Yellow Leaf Curl Virus",
    ],
    "Rice": [
        "Bacterial Leaf Blight",
        "Bacterial Leaf Streak",
        "Bacterial Panicle Blight",
        "Rice Blast",
        "Brown Spot",
        "Dead Heart",
        "Downy Mildew",
        "Healthy Rice",
        "Rice Hispa",
        "Tungro",
    ],
    "Orange": [
        "Black Spot",
        "Canker",
        "Greening (HLB)",
        "Healthy Orange",
        "Scab",
    ],
}

# Data directory names (lowercase, must match data/<dir>/).
CROP_DATA_DIRS = {crop: crop.lower() for crop in CROP_CLASSES}

# File-safe directory names for each disease class.
# These MUST match the subdirectory names created by setup.py inside data/<crop>/.
def class_to_dirname(name):
    """Convert display name to filesystem-safe directory name."""
    return name.replace(" ", "_").replace("(", "").replace(")", "")

# Flat mapping for convenience: (crop, class_index) -> display_name
def get_disease_name(crop, class_index):
    return CROP_CLASSES[crop][class_index]

def get_num_classes(crop):
    return len(CROP_CLASSES[crop])

def get_crop_data():
    """Return structured crop info dict compatible with app.py/train.py loops."""
    return {
        crop: {
            "classes": classes,
            "num_classes": len(classes),
            "data_dir": CROP_DATA_DIRS[crop],
        }
        for crop, classes in CROP_CLASSES.items()
    }
