# AGENTS.md — CropGuard

## Architecture

Multi-crop leaf disease classifier: **shared ResNet50 backbone** + **per-crop Keras classifier heads**.

```
Image → ResNet50 (1000-dim features) → crop-specific Dense classifier → disease name
```

- `crop_config.py` — single source of truth for disease class names per crop. No treatment data lives here.
- Disease treatment/symptoms/prevention come from the **LLM** (Groq API via `obtener_tratamiento_llm`), not a static DB.
- Per-crop models: `models/cropguard_{tomato|rice|orange}_classifier.keras`
- Shared backbone: `models/cropguard_resnet50.pth`

## Setup order

```
python setup.py              # download datasets → data/<crop>/
python train.py              # train per-crop classifiers → models/
streamlit run app.py         # launch web app
```

`setup.py` uses `kagglehub` for Kaggle datasets. Rice download is ~780 MB.

## Critical Streamlit rules

1. **`st.set_page_config` MUST be the very first Streamlit call** in the file — before any `@st.cache_resource` decorators, `st.columns`, or any other `st.*` call. Getting this wrong causes silent script-thread crashes with no Python traceback.

2. **Do not use `use_container_width`** — it was removed in Streamlit ≥1.59. Just `st.image(img, caption="...")`.

3. **Import heavy packages at module top level** (torch, keras, torchvision). Keras import inside `@st.cache_resource` functions or lazy-imported in button handlers silently crashes the script runner thread. Import everything before the first `st.*` call after `set_page_config`.

## Environment

- `KERAS_BACKEND=torch` (set at top of every script before imports)
- Keras 3.15.0, torch 2.6.0+cu124, streamlit 1.59.2
- Windows paths — use `os.path.join`, not `/` literals
- CUDA available but `train.py` falls back gracefully

## Optional packages

`app.py` degrades gracefully if these are missing:
- `openai` — needed for LLM treatment advice (Groq API)
- `gtts` — needed for TTS audio

Secrets file: `.streamlit/secrets.toml` with `GROQ_API_KEY = "..."`

## Data directory structure

```
data/
  tomato/   Bacterial_spot/  Early_blight/  Healthy/  Late_blight/  Leaf_Mold/
            Mosaic_virus/  Septoria_spot/  Spider_mites/  Target_Spot/  Yellow_Curl/
  rice/     (10 class subdirs)
  orange/   (5 class subdirs)
```

Class subdirectory names in `data/` must match the names used by `torchvision.datasets.ImageFolder` — which sorts alphabetically. `crop_config.CROP_CLASSES` lists them in this same order.
