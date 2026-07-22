"""
CropGuard Dataset Setup Script
Downloads and organizes datasets for Tomato, Rice, and Orange crops.

Usage:
    python setup.py              # Setup all crops
    python setup.py --tomato     # Tomato only
    python setup.py --tomato --rice  # Tomato + Rice
    python setup.py --force      # Re-download even if data exists
"""
import os, sys, shutil, argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = Path.cwd()
DATA_DIR = REPO_ROOT / "src" / "data"


TOMATO_MAP = {
    "Tomato_Bacterial_spot":                      "Bacterial_spot",
    "Tomato_Early_blight":                        "Early_blight",
    "Tomato_healthy":                             "Healthy",
    "Tomato_Late_blight":                         "Late_blight",
    "Tomato_Leaf_Mold":                           "Leaf_Mold",
    "Tomato_Septoria_leaf_spot":                  "Septoria_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spider_mites",
    "Tomato__Target_Spot":                        "Target_Spot",
    "Tomato__Tomato_mosaic_virus":                "Mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus":      "Yellow_Curl",
}

RICE_MAP = {
    "bacterial_leaf_blight":   "Bacterial_Leaf_Blight",
    "bacterial_leaf_streak":   "Bacterial_Leaf_Streak",
    "bacterial_panicle_blight": "Bacterial_Panicle_Blight",
    "blast":                   "Rice_Blast",
    "brown_spot":              "Brown_Spot",
    "dead_heart":              "Dead_Heart",
    "downy_mildew":            "Downy_Mildew",
    "hispa":                   "Rice_Hispa",
    "normal":                  "Healthy_Rice",
    "tungro":                  "Tungro",
}

ORANGE_MAP = {
    "Black spot":    "Black_Spot",
    "black spot":    "Black_Spot",
    "canker":        "Canker",
    "greening":      "Greening_HLB",
    "healthy":       "Healthy_Orange",
    "scab":          "Scab",
}


def banner(msg):
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


def count_images(path):
    if not os.path.isdir(path):
        return 0
    return sum(
        1 for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    )


def copy_rename_folder(src, dst, rename_map):
    if not os.path.isdir(src):
        print(f"  ERROR: source directory not found: {src}")
        return

    # Build index of source folders once
    src_folders = {
        name: os.path.join(src, name)
        for name in os.listdir(src)
        if os.path.isdir(os.path.join(src, name))
    }

    for pattern, target in rename_map.items():
        pattern_lower = pattern.lower()
        matched = None
        for folder_name, folder_path in src_folders.items():
            if folder_name.lower() == pattern_lower or (
                pattern_lower in folder_name.lower()
                and folder_name.lower().startswith(pattern_lower.split("_")[0].lower())
            ):
                matched = folder_path
                break

        # Fallback: substring match
        if matched is None:
            for folder_name, folder_path in src_folders.items():
                if pattern_lower in folder_name.lower():
                    matched = folder_path
                    break

        if matched is None:
            print(f"  WARNING: no folder matching '{pattern}' found in {src}")
            continue

        dst_path = os.path.join(dst, target)
        if not os.path.exists(dst_path):
            shutil.copytree(matched, dst_path)
            n = count_images(dst_path)
            print(f"  {os.path.basename(matched):45s} -> {target:30s} ({n} images)")
        else:
            n = count_images(dst_path)
            print(f"  {os.path.basename(matched):45s} -> {target:30s} (EXISTS, {n} images)")


def _flatten_nested_dirs(parent_dir):
    """Move contents of a single nested subdirectory up to parent_dir.

    Some zip archives create a root folder (e.g. 'Orange Fruit Diseases Dataset/')
    containing the actual class folders. This flattens that structure.
    """
    for _ in range(3):  # max nesting depth
        entries = [e for e in os.listdir(parent_dir)
                   if os.path.isdir(os.path.join(parent_dir, e))]
        if len(entries) == 1:
            nested = os.path.join(parent_dir, entries[0])
            print(f"  Flattening nested directory: {entries[0]}")
            for item in os.listdir(nested):
                src = os.path.join(nested, item)
                dst = os.path.join(parent_dir, item)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            shutil.rmtree(nested)
        else:
            break


# ---- TOMATO ------------------------------------------------------------
def setup_tomato(force=False):
    dest = os.path.join(DATA_DIR, "tomato")
    if not force and os.path.isdir(dest) and all(
        count_images(os.path.join(dest, d)) > 0 for d in TOMATO_MAP.values()
    ):
        total = sum(count_images(os.path.join(dest, d)) for d in TOMATO_MAP.values())
        print(f"  Tomato data already exists ({total} images). Use --force to re-download.")
        return

    os.makedirs(dest, exist_ok=True)

    import kagglehub
    cache_root = kagglehub.dataset_download("emmarex/plantdisease")

    src_dir = os.path.join(cache_root, "PlantVillage")
    if not os.path.isdir(src_dir):
        src_dir = os.path.join(cache_root)
        subdirs = [d for d in os.listdir(src_dir)
                   if os.path.isdir(os.path.join(src_dir, d)) and d.startswith("Tomato")]
        if not subdirs:
            for root, dirs, _ in os.walk(src_dir):
                if any(d.startswith("Tomato") for d in dirs):
                    src_dir = root
                    break

    print(f"  Source: {src_dir}")
    copy_rename_folder(src_dir, dest, TOMATO_MAP)


# ---- RICE --------------------------------------------------------------
def setup_rice(force=False):
    dest = os.path.join(DATA_DIR, "rice")
    if not force and os.path.isdir(dest) and all(
        count_images(os.path.join(dest, d)) > 0 for d in RICE_MAP.values()
    ):
        total = sum(count_images(os.path.join(dest, d)) for d in RICE_MAP.values())
        print(f"  Rice data already exists ({total} images). Use --force to re-download.")
        return

    os.makedirs(dest, exist_ok=True)

    import kagglehub
    cache_root = kagglehub.dataset_download("yashzanwar/rice-leaf-disease-classification")

    src_dir = None
    for root, dirs, _ in os.walk(cache_root):
        if "train_images" in dirs:
            src_dir = os.path.join(root, "train_images")
            break
    if src_dir is None:
        subdirs = [d for d in os.listdir(cache_root)
                   if os.path.isdir(os.path.join(cache_root, d)) and not d.startswith(".")]
        if subdirs:
            src_dir = cache_root

    if src_dir is None:
        print(f"  ERROR: could not find training images in {cache_root}")
        return

    print(f"  Source: {src_dir}")
    copy_rename_folder(src_dir, dest, RICE_MAP)


# ---- ORANGE ------------------------------------------------------------
def setup_orange(force=False):
    dest = os.path.join(DATA_DIR, "orange")
    if not force and os.path.isdir(dest) and all(
        count_images(os.path.join(dest, d)) > 0 for d in ORANGE_MAP.values()
    ):
        total = sum(count_images(os.path.join(dest, d)) for d in ORANGE_MAP.values())
        print(f"  Orange data already exists ({total} images). Use --force to re-download.")
        return

    os.makedirs(dest, exist_ok=True)

    # Step 1: Check if the user already downloaded the zip
    zip_path = os.path.join(REPO_ROOT, "orange_dataset.zip")
    extracted_dir = None

    if os.path.exists(zip_path):
        print(f"  Found orange_dataset.zip in project root, extracting...")
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        os.unlink(zip_path)

        # Flatten nested directory if the zip creates one (e.g. "Orange Fruit Diseases Dataset/")
        _flatten_nested_dirs(dest)

        # Find the extracted content
        extra_dirs = []
        for root, dirs, _ in os.walk(dest):
            leaf_dirs = [d for d in dirs
                         if d.lower() in {"black spot", "canker", "greening", "healthy", "scab"}]
            if leaf_dirs:
                extracted_dir = root
                break
            extra_dirs = [d for d in dirs if d.lower() not in {"black spot", "canker", "greening", "healthy", "scab", "__macosx"}]
        else:
            # No class dirs found — check if the zip created a single parent folder
            if not extracted_dir and extra_dirs:
                print(f"  WARNING: could not find expected class folders in extracted zip.")
                print(f"  Please extract manually to: {dest}")
                return

    # Step 2: Check if already extracted manually
    if extracted_dir is None:
        for root, dirs, _ in os.walk(dest):
            leaf_dirs = [d for d in dirs
                         if d.lower() in {"black spot", "canker", "greening", "healthy", "scab"}]
            if leaf_dirs:
                extracted_dir = root
                break

    # Step 3: If nothing found, instruct the user
    if extracted_dir is None:
        print(f"\n  ----------------------------------------------")
        print(f"  The Orange Fruit Diseases dataset")
        print(f"  requires manual download (Cloudflare blocks")
        print(f"  automated downloads from Mendeley).")
        print(f"")
        print(f"  1. Open this link in your browser:")
        print(f"     https://data.mendeley.com/datasets/6szsnpypdd/1")
        print(f"")
        print(f"  2. Click 'Download All' and save as:")
        print(f"     {os.path.join(REPO_ROOT, 'orange_dataset.zip')}")
        print(f"")
        print(f"  3. Run this script again: python setup.py --orange")
        print(f"  ----------------------------------------------")
        return

    print(f"  Source: {extracted_dir}")
    copy_rename_folder(extracted_dir, dest, ORANGE_MAP)


# ---- MAIN --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CropGuard Dataset Setup")
    parser.add_argument("--tomato", action="store_true", help="Setup tomato dataset")
    parser.add_argument("--rice", action="store_true", help="Setup rice dataset")
    parser.add_argument("--orange", action="store_true", help="Setup orange dataset")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    all_crops = not (args.tomato or args.rice or args.orange)
    banner("CropGuard - Dataset Setup")

    os.makedirs(DATA_DIR, exist_ok=True)

    if all_crops or args.tomato:
        banner("TOMATO (PlantVillage via Kaggle)")
        try:
            setup_tomato(force=args.force)
        except Exception as e:
            print(f"  FAILED: {e}")

    if all_crops or args.rice:
        banner("RICE (Rice Leaf Disease via Kaggle)")
        try:
            setup_rice(force=args.force)
        except Exception as e:
            print(f"  FAILED: {e}")

    if all_crops or args.orange:
        banner("ORANGE (Citrus Leaves via Kaggle/Mendeley)")
        try:
            setup_orange(force=args.force)
        except Exception as e:
            print(f"  FAILED: {e}")
            print(f"  Manual download: https://data.mendeley.com/datasets/6szsnpypdd/1")

    banner("SUMMARY")
    for crop in ["tomato", "rice", "orange"]:
        path = os.path.join(DATA_DIR, crop)
        if os.path.isdir(path):
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            total = sum(count_images(os.path.join(path, d)) for d in subdirs)
            print(f"  {crop.capitalize():10s}: {total:>6,} images in {len(subdirs)} classes")
        else:
            print(f"  {crop.capitalize():10s}: not present")

    banner("DONE")
    print("Run: python train.py   to train classifiers")
    print("Run: streamlit run app.py  to launch the app")


if __name__ == "__main__":
    main()
