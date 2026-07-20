# CropGuard

AI-powered plant disease classifier for Android. Snap a photo of a leaf, get an instant diagnosis with treatment advice in English, Spanish, or Valencian.

Built as a capstone project for [Samsung Innovation Campus](https://innovationcampus.samsung.com/).

## Features

- **Multi-crop diagnosis** — 25 diseases across Tomato (10), Rice (10), and Orange (5)
- **Real-time inference** — fine-tuned ResNet50 runs locally on-device (or via backend API)
- **Trilingual support** — treatment cards in English, Spanish, and Valencian
- **AI follow-up chat** — ask follow-up questions about a diagnosis (Gemini free tier)
- **Android app** — Kotlin + Jetpack Compose with camera and gallery capture
- **Web demo** — Streamlit app for quick browser-based testing

## Architecture

```
Leaf photo
  ↓
ResNet50 (ImageNet-pretrained, layer4 fine-tuned)
  ↓
MLP head: 2048 → 256 → 128 → num_classes
  ↓
Softmax → disease class → treatment card
```

Each crop has its own independently trained model (~92 MB). Two-phase fine-tuning per crop: head warmup (backbone frozen) then full fine-tuning (layer4 unfrozen).

## Quick Start

### Prerequisites

- Python 3.12+
- [Kaggle API credentials](https://www.kaggle.com/docs/api) (for dataset download)
- GPU recommended for training, CPU sufficient for inference

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download datasets

```bash
python -m cropguard.setup              # all crops
python -m cropguard.setup --tomato     # single crop
```

Orange requires [manual download](https://data.mendeley.com/datasets/6szsnpypdd/1) (Cloudflare blocks automation). Save as `orange_dataset.zip` in the repo root, then:

```bash
python -m cropguard.setup --orange
```

### 3. Train models

```bash
python -m cropguard.train                              # all crops
python -m cropguard.train --crop rice --epochs 25      # single crop
```

Models saved to `models/cropguard_<crop>_model.pth`.

### 4. Launch

```bash
# Web app
streamlit run src/cropguard/app.py

# Backend API (for Android app)
uvicorn cropguard.server:app --host 0.0.0.0 --port 8000

# CLI inference
python -m cropguard.predict_worker path/to/leaf.jpg Tomato
```

## Project Structure

```
CropGuard/
├── src/cropguard/           # Python package
│   ├── crop_config.py       # disease classes per crop
│   ├── model_def.py         # ResNet50 architecture + transforms
│   ├── train.py             # two-phase fine-tuning loop
│   ├── app.py               # Streamlit web demo
│   ├── server.py            # FastAPI backend
│   ├── predict_worker.py    # CLI inference
│   ├── setup.py             # dataset download/setup
│   ├── llm_advice.py        # Gemini AI + static fallback
│   └── treatments.json      # 25 diseases × 3 languages
├── android/                 # Kotlin + Jetpack Compose app
├── tests/                   # pytest unit + e2e tests
├── scripts/                 # helper scripts
└── docs/                    # capstone templates
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | ResNet50 (torchvision), fine-tuned per crop |
| Training | PyTorch, scikit-learn, matplotlib |
| Web demo | Streamlit |
| Backend API | FastAPI + uvicorn |
| Android | Kotlin, Jetpack Compose, Material 3, Retrofit |
| AI chat | Google Gemini 2.0 Flash (free tier) |
| CI/CD | GitHub Actions (Python + Android jobs) |

## Android App

The Android app communicates with the FastAPI backend:

- **Emulator**: `http://10.0.2.2:8000` (default)
- **Physical device**: use your PC's LAN IP

```bash
# Build APK
android\gradlew.bat -p android assembleDebug

# Run tests
android\gradlew.bat -p android testDebugUnitTest
```

Requires JDK 17. Use Android Studio's bundled JBR:

```bash
set JAVA_HOME=C:\Program Files\Android\Android Studio\jbr
set ANDROID_HOME=%LOCALAPPDATA%\Android\Sdk
```

## Testing

```bash
python -m pytest tests/ -v          # all tests
python -m pytest tests/unit -v      # unit tests only
python -m pytest tests/e2e -v       # end-to-end (spawns real server)
```

Tests run without real models or API keys — random-weight models are generated in a temp directory.

## AI Follow-up Chat (Optional)

Set the `GEMINI_API_KEY` environment variable to enable AI-powered follow-up questions:

```bash
# Windows
setx GEMINI_API_KEY "your-key-here"

# Linux/macOS
export GEMINI_API_KEY="your-key-here"
```

Get a free key at [aistudio.google.com](https://aistudio.google.com/apikey). Without it, the app returns pre-written treatment advice (no AI).

## License

Samsung Innovation Campus capstone project.
