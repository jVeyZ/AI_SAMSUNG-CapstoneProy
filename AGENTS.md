# AGENTS.md — CropGuard

Multi-crop leaf-disease classifier (Samsung Innovation Campus capstone). Pure-torch pipeline: one fine-tuned ResNet50 per crop (no Keras/TensorFlow anywhere).

```
image → ResNet50 (ImageNet-normalized input, layer4 fine-tuned, fc = MLP head 2048→256→128→classes) → softmax → class index → display name
```

- `crop_config.py` — single source of truth: `CROP_CLASSES` display names; `class_to_dirname()` maps display name → data-dir name.
- `model_def.py` — shared contract between training and inference: `build_model(num_classes)` (ResNet50 + MLP head in `fc`), `TRAIN_TRANSFORM`/`EVAL_TRANSFORM` (include ImageNet mean/std normalization — required, the pretrained backbone is off-distribution without it), `model_path(crop)`. Change the head architecture ONLY here.
- Treatment/symptom text comes from a Groq LLM call at runtime (`obtener_tratamiento_llm` in app.py), not a database. LLM prompts and TTS are Spanish; the UI is English.
- `data/`, `models/`, `results/` are gitignored — a fresh clone has no datasets or models; regenerate with the commands below.
- Models: `models/cropguard_<crop>_model.pth` — full fine-tuned state dict (backbone + head), ~92 MB per crop.

## Commands (run in this order)

```
python setup.py        # download datasets → data/<crop>/   [--tomato --rice --orange --force]
python train.py        # fine-tune all crops → models/      [--crop rice --epochs 25 --head-epochs 8 --ft-lr 1e-4 --batch-size 32 --workers 4 --no-early-stop --no-weights]
streamlit run app.py   # launch app
```

- `setup.py` uses `kagglehub` (needs Kaggle API credentials in `~/.kaggle/kaggle.json`); rice download is ~780 MB.
- **Orange cannot be auto-downloaded** (Cloudflare blocks Mendeley): download https://data.mendeley.com/datasets/6szsnpypdd/1 manually, save as `orange_dataset.zip` in the repo root, then rerun `python setup.py --orange`.
- `train.py` skips crops whose data dir is missing. Per crop it runs two phases: head warmup (backbone frozen) then fine-tuning (layer4 unfrozen, head LR = 10× `--ft-lr`). It checkpoints the best weights to `model_path(crop)` on every val improvement, so an interrupted run still leaves a usable model — but figures/JSON are only written on completion.
- `train.py` writes per-crop figures to `results/<crop>/` (`history.png` with fine-tune boundary, `confusion_matrix.png`, `confusion_matrix_normalized.png`, `class_metrics.png`, `confidence_histogram.png`, `misclassified_samples.png`, `class_distribution.png`) and enriches `results/training_results.json` with `top2_accuracy`, `macro_f1`, `weighted_f1`, `per_class` metrics, `head_epochs_ran`, `epochs_ran`, `best_epoch`, `finetuned`, `unfrozen`.
- No requirements.txt, no test suite, no CI. Deps: torch, torchvision, streamlit, scikit-learn, matplotlib, pillow, kagglehub; optional `openai` + `gtts` (app degrades gracefully without them — both are absent in this env and the app still runs).
- Smoke checks: `python predict_worker.py <image> <Crop>` prints a JSON prediction to stdout. For UI changes: `AppTest.from_file("app.py")` from `streamlit.testing.v1` runs the script headlessly.

## Environment

- Verified working env: torch 2.6.0+cu124, streamlit 1.59.2, Windows, RTX 3050 8GB. `train.py` trains on GPU; `app.py`/`predict_worker.py` inference is CPU-only.
- Keras/TensorFlow are NOT needed (removed when the pipeline went pure-torch). The old `KERAS_BACKEND=torch` requirement is gone.
- LLM features need `.streamlit/secrets.toml` with `GROQ_API_KEY = "..."` (Groq via OpenAI-compatible client, model `llama-3.3-70b-versatile`).

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
