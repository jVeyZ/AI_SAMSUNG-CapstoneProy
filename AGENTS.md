# AGENTS.md — CropGuard

Multi-crop leaf-disease classifier (Samsung Innovation Campus capstone). Pure-torch pipeline: one fine-tuned ResNet50 per crop (no Keras/TensorFlow anywhere).

```
image → ResNet50 (ImageNet-normalized input, layer4 fine-tuned, fc = MLP head 2048→256→128→classes) → softmax → class index → display name
```

- `crop_config.py` — single source of truth: `CROP_CLASSES` display names; `class_to_dirname()` maps display name → data-dir name.
- `model_def.py` — shared contract between training and inference: `build_model(num_classes)` (ResNet50 + MLP head in `fc`), `TRAIN_TRANSFORM`/`EVAL_TRANSFORM` (include ImageNet mean/std normalization — required, the pretrained backbone is off-distribution without it), `model_path(crop)`. Change the head architecture ONLY here.
- `treatments.json` — static treatment advice (explanation/symptoms/treatment/prevention) for all 25 diseases × 3 languages (`en`/`es`/`va`). This is the DEFAULT advice everywhere.
- `llm_advice.py` — shared helper: `get_static_treatment()` + `ask_followup()` (Gemini free tier, `gemini-2.0-flash`, env var `GEMINI_API_KEY`, google-genai package). AI answers use the static content as grounding context; without a key it returns the static fallback, never an error.
- `server.py` — FastAPI backend for the Android app: `GET /health`, `GET /crops`, `POST /predict` (multipart), `GET /treatment/{crop}/{disease}?lang=`, `POST /chat`. Models load lazily per crop on CPU; `CROPGUARD_MODELS_DIR` env var overrides the models dir (tests inject random-weight models).
- `android/` — Kotlin + Jetpack Compose app (Material 3). Talks to `server.py`; i18n EN/ES/Valencian via an in-memory dictionary (deliberately NOT res/values-XX — locale recreation would wipe ViewModel state).
- `data/`, `models/`, `results/` are gitignored — a fresh clone has no datasets or models; regenerate with the commands below.
- Models: `models/cropguard_<crop>_model.pth` — full fine-tuned state dict (backbone + head), ~92 MB per crop.

## Commands (run in this order)

```
python setup.py          # download datasets → data/<crop>/   [--tomato --rice --orange --force]
python train.py          # fine-tune all crops → models/      [--crop rice --epochs 25 --head-epochs 8 --ft-lr 1e-4 --batch-size 32 --workers 4 --no-early-stop --no-weights]
streamlit run app.py     # launch web app
uvicorn server:app --port 8000   # launch backend API for the Android app
python -m pytest tests/unit tests/e2e   # run test suites
android\gradlew.bat -p android testDebugUnitTest assembleDebug   # Android tests + APK (needs JDK 17, see below)
```

- `setup.py` uses `kagglehub` (needs Kaggle API credentials in `~/.kaggle/kaggle.json`); rice download is ~780 MB.
- **Orange cannot be auto-downloaded** (Cloudflare blocks Mendeley): download https://data.mendeley.com/datasets/6szsnpypdd/1 manually, save as `orange_dataset.zip` in the repo root, then rerun `python setup.py --orange`.
- `train.py` skips crops whose data dir is missing. Per crop it runs two phases: head warmup (backbone frozen) then fine-tuning (layer4 unfrozen, head LR = 10× `--ft-lr`). It checkpoints the best weights to `model_path(crop)` on every val improvement, so an interrupted run still leaves a usable model — but figures/JSON are only written on completion.
- `train.py` writes per-crop figures to `results/<crop>/` (`history.png` with fine-tune boundary, `confusion_matrix.png`, `confusion_matrix_normalized.png`, `class_metrics.png`, `confidence_histogram.png`, `misclassified_samples.png`, `class_distribution.png`) and enriches `results/training_results.json` with `top2_accuracy`, `macro_f1`, `weighted_f1`, `per_class` metrics, `head_epochs_ran`, `epochs_ran`, `best_epoch`, `finetuned`, `unfrozen`.
- Deps in `requirements.txt` (+ `requirements-dev.txt` for tests). CI installs CPU-only torch wheels (`--index-url https://download.pytorch.org/whl/cpu`) — do the same on Linux CI-like envs.
- Smoke checks: `python predict_worker.py <image> <Crop>` prints a JSON prediction to stdout. For UI changes: `AppTest.from_file("app.py")` from `streamlit.testing.v1` runs the script headlessly.
- CI: `.github/workflows/ci.yml` — job `python` (pytest unit + e2e) and job `android` (gradle tests + assembleDebug). No real models/data/API keys in CI: tests build random-weight models into a temp dir and exercise the Gemini fallback path.

## Environment

- Verified working env: torch 2.6.0+cu124, streamlit 1.59.2, Windows, RTX 3050 8GB. `train.py` trains on GPU; `app.py`/`server.py`/`predict_worker.py` inference is CPU-only.
- Keras/TensorFlow are NOT needed (removed when the pipeline went pure-torch). The old `KERAS_BACKEND=torch` requirement is gone.
- AI follow-up answers need env var `GEMINI_API_KEY` (free key from aistudio.google.com). Without it, `llm_advice.ask_followup()` returns the static treatment as fallback — tests rely on this.
- Android builds need **JDK 17** — the system `java` on this machine is 1.8 and will NOT work. Use Android Studio's bundled JBR: set `JAVA_HOME=C:\Program Files\Android\Android Studio\jbr` and `ANDROID_HOME=%LOCALAPPDATA%\Android\Sdk` before running gradlew.
- Android demo networking: emulator reaches the host via `http://10.0.2.2:8000` (default in `BuildConfig.BASE_URL`); a physical phone on the same Wi-Fi needs the PC's LAN IP there instead.

## app.py / Streamlit gotchas

- `st.set_page_config` must come before any st UI call (verified on 1.59.2).
- `use_container_width` is deprecated (warning only, not removed) in streamlit 1.59.2 — use `width="stretch"` or omit it, as app.py does.
- Keep torch/model_def imports at module top level in app.py; lazy-importing heavy frameworks inside handlers has caused silent script-thread crashes (no traceback). The lazy `matplotlib` import in the diagnosis branch is fine.

## Class names ↔ data dirs (fragile)

`ImageFolder` numbers classes alphabetically over `data/<crop>/` subdirs; `CROP_CLASSES[crop]` must list display names in that same order. `train.py`, `app.py`, and `predict_worker.py` all depend on this alignment — keep it when adding crops or renaming classes.

## Don't-touch quirks

- `save_model.py` is legacy tomato-only code superseded by `train.py`: it hardcodes `.cuda()` (crashes without GPU), requires Keras (no longer a project dep), and writes obsolete artifacts (`cropguard_model.keras`, flat-schema `results/training_results.json`). Don't run it.
- In `train.py`, train and val/test come from two separate `ImageFolder` instances split with the same seeded generator — keep the seeds identical or the splits diverge.
- `train.py`'s driver is wrapped in `main()` + `if __name__ == "__main__"` guard — required on Windows so `num_workers > 0` DataLoader processes don't re-run training on spawn. Don't move driver code back to module top level.
