# AI Course — Team Project Final Report

ⓒ2019 SAMSUNG. All rights reserved. Samsung Electronics Corporate Citizenship Office holds the copyright of this document. This document is a literary property protected by copyright law so reprint and reproduction without permission are prohibited. To use this document other than the curriculum of Samsung Innovation Campus, you must receive written consent from copyright holder.

| **CropGuard: AI-Powered Multi-Crop Disease Classifier** |
| :-----------------------------------------------------: |

**22/07/2026**

**Team Name: CropGuard**

---

# Abstract

Crop diseases destroy 20–40% of global harvests annually before reaching markets, with smallholder farmers in low-income regions bearing the heaviest losses due to limited access to expert phytosanitary diagnosis. CropGuard addresses this gap with an end-to-end deep learning pipeline that diagnoses 25 diseases across Tomato, Rice, and Orange from a single leaf photo—delivering sub-second inference on commodity hardware. The system couples a fine-tuned ResNet50 classifier (96.25–99.00% test accuracy per crop) with a multi-platform ecosystem: a FastAPI backend, a Streamlit web demo, and a fully trilingual Kotlin/Jetpack Compose Android application. Treatment recommendations are available in English, Spanish, and Valencian via 75 curated expert advisory cards, while an optional AI chat layer (Gemini or OpenCode) answers farmer follow-up questions with ecological, actionable advice. The entire pipeline runs on free-tier compute—training requires GPU, inference runs on CPU—and continuous integration ensures 88 automated tests pass on every push.

---

# Introduction

## Objective

Build a lightweight, multi-crop leaf-disease classifier that:

1. Accepts a smartphone photo of any leaf from Tomato, Rice, or Orange.
2. Returns the disease name, confidence score, and a localized treatment plan within 1 second.
3. Provides a trilingual Android app that works offline treatment-wise and sends photos to a backend for AI inference.
4. Answers follow-up agronomic questions via free-tier AI (no paid API keys required).

## Motivation

Agriculture sustains 60% of the world's population, yet plant pathology expertise is scarce—there is roughly one agronomist per 5,000 farmers in developing nations. Existing tools like PlantVillage offer classification but stop at the diagnosis; they rarely include actionable treatment advice, and none combine on-device usability with free AI-powered Q&A in minority languages (Valencian, a dialect of Catalan with 2.4 million speakers). CropGuard fills exactly this void: diagnosis → treatment → follow-up conversation, all in the farmer's language, all on a phone that costs under €100.

## Product Summary

CropGuard is three things in one repository:

1. **A fine-tuned classification pipeline** (PyTorch, ResNet50) that identifies 25 diseases with ≥96% accuracy.
2. **A ready-to-ship Android app** (Jetpack Compose) where a farmer snaps a photo, picks a crop, and receives a diagnosis card with an expandable confidence breakdown, a structured treatment plan (symptoms → treatment → prevention), and a chat window for asking the AI follow-up questions.
3. **A zero-cost AI layer** using free-tier LLMs (OpenCode's qwen3.5-plus and Google Gemini 2.0 Flash) that translates expert knowledge into plain-language advice in English, Spanish, or Valencian—with text-to-speech playback for farmers who prefer listening over reading.

## Background

Crop diseases are estimated to cause over $220 billion in annual economic losses globally. Common practice on small farms is either unaided visual inspection (error-prone) or reliance on extension services (unavailable at scale). The relevant AI domain is fine-grained visual classification using convolutional neural networks pre-trained on ImageNet—a proven technique that achieves near-expert performance without requiring millions of domain-specific training images.

The three target crops cover distinct socioeconomic profiles: Tomato (10 disease classes, high-value greenhouse crop in Europe), Rice (10 classes, staple food for half of humanity), and Orange (5 classes, cash crop affected by HLB/"citrus greening," a pandemic-level threat). Each crop demands its own classifier because symptom expression varies wildly across species.

Our approach sits at the intersection of transfer learning, mobile health-for-plants, and low-resource NLP in Iberian Romance languages.

## Related Work

- **PlantVillage (2015–2019):** Pioneered CNN-based plant disease classification with 54,306 images across 14 crops. Our work extends this by adding treatment content and an LLM chat layer.
- **Mohanty et al. (2016):** Showed that deep learning can distinguish 26 diseases at 99.35% accuracy. We adopt their transfer learning philosophy but fine-tune only layer4 instead of the full backbone, reducing training time by 60% with negligible accuracy loss.
- **AgriBot, Plantix, etc.:** Commercial smartphone apps exist but require subscriptions, lack chat capabilities, and support only major world languages—none offer Valencian/Catalan.

**Novelty:** CropGuard is, to our knowledge, the first open-source plant-disease app combining fine-tuned ResNet50 with free-tier LLM-powered agronomic advice in three languages including a minority language, plus a production-ready Android APK buildable in CI.

---

# Product

## Market

**Primary persona:** A Valencian Community citrus farmer (age 45–65) who owns 2–5 hectares, photographs fruit with a Xiaomi or Samsung budget phone, and reads Spanish or Valencian but not English. They need a free diagnostic tool that explains results in simple terms and suggests treatments they can obtain locally (neem oil, copper sulfate, biological control agents).

**Secondary persona:** A rice-extension officer in South Asia who handles ~200 inquiries per week and uses the app to triage cases—the chat feature lets them ask for specific recommendations while the language toggle switches to English for international reports.

**Tertiary persona:** A university student in an AI course using the public codebase as a reference implementation for a full-stack ML project (backend, web, mobile, CI).

## Product Description & Usage

### Android App Flow

```
Home Screen (CaptureScreen)
  ├── [Crop selector: Tomato / Rice / Orange]
  ├── [Image area: camera snap or gallery pick]
  ├── [Analyze button]
  └── Language menu (EN / ES / VA) + Settings gear (server URL, AI provider)

      ↓ (analyze → sends photo to FastAPI backend)

Result Screen
  ├── Card: Disease name, crop, confidence bar
  │    └── [Show all classes ▼] → sorted probability list
  ├── TreatmentCard: symptoms / treatment / prevention
  │    └── [▶ Play] TTS button (reads full treatment aloud)
  ├── Chat: follow-up Q&A with AI (OpenCode or Gemini)
  │    ├── ChatBubble(user): "Can I use neem oil?"
  │    ├── ChatBubble(AI): provider name + [▶ Play] + markdown answer
  │    └── Typing indicator (pulsing dots) while waiting
  └── Text field + Send button
```

### Web App (Streamlit)

Same flow but browser-based: file upload, crop dropdown, diagnosis visualization with per-class bar charts, and an optional Gemini/OpenCode chat panel. Primarily for demonstration and teacher review.

### Backend API (FastAPI)

REST endpoints:
- `GET /health` — Liveness check
- `GET /crops` — List crops and disease classes
- `POST /predict` — Upload image + crop name, receive JSON with disease, confidence, and full probability distribution
- `GET /treatment/{crop}/{disease}?lang=` — Static treatment card in requested language
- `POST /chat` — AI follow-up (body: crop, disease, question, lang, provider)

### AI Integration Point

The AI enters at two levels:

1. **Static treatment** (`treatments.json`): 25 diseases × 3 languages = 75 expert-written, curated treatment cards. Always available offline, zero-cost.
2. **Dynamic follow-up** (`llm_advice.py`): The server sends a carefully crafted prompt containing the static treatment JSON plus the farmer's question. The LLM responds in the farmer's language with concise (~150 words), ecological, practical advice. Without an API key, the static treatment is returned as a fallback.

---

# Results

## Data

| Crop   | Classes | Source | Train | Val | Test | Total |
|--------|---------|--------|-------|-----|------|-------|
| Tomato | 10      | Kaggle (kagglehub) | 12,008 | 2,401 | 1,602 | 16,011 |
| Rice   | 10      | Kaggle (kagglehub) | 7,805 | 1,561 | 1,041 | 10,407 |
| Orange | 5       | Mendeley (manual) | 525 | 105 | 70 | 700 |

The Tomato and Rice datasets come from public Kaggle repositories (autodownload via `kagglehub`). The Orange dataset from Mendeley requires manual download due to Cloudflare anti-bot protection, saved as `orange_dataset.zip`. All images were split 60/20/20 with a fixed random seed to ensure reproducibility. Orange is the smallest dataset (5 classes, 700 images) because its classes are more visually distinct, while Tomato has the largest (16,011) to handle high intra-class variability across 10 diseases.

**Preprocessing:** All images resized to 224×224, normalized with ImageNet statistics (µ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]). Training augmentations: random horizontal/vertical flip (p=0.5), ±15° rotation, mild color jitter. Validation and test use clean evaluation transforms only.

## Modeling

**Architecture:** ResNet50 (torchvision, IMAGENET1K_V2 weights) with a 3-layer MLP classifier head replacing the original fully-connected layer:

```
image (224×224×3)
  → ResNet50 backbone (2048-dim feature vector)
  → FC(2048 → 256) → ReLU → Dropout(0.35)
  → FC(256 → 128) → ReLU → Dropout(0.35)
  → FC(128 → num_classes)
  → Softmax
```

**Independent models:** One model per crop—each stored as `cropguard_<crop>_model.pth` (~92 MB). This avoids catastrophic interference between crops and allows updating one crop's classifier without retraining the others.

**Two-phase training (PyTorch, one RTX 3050 8 GB):**

- **Phase 1 — Head warmup (8-15 epochs):** Backbone frozen. Only the MLP head learns. Adam optimizer, learning rate 1×10⁻³, batch size 32. This gives the head a meaningful starting point before unfreezing.
- **Phase 2 — Fine-tuning (7-10 epochs):** Layer4 of ResNet50 unfrozen (conv4_x blocks). Backbone LR = 1×10⁻⁴, head LR = 1×10⁻³ (10× multiplier). Best weights checkpointed on validation improvement. Early stopping with 5-epoch patience.

**Loss:** Cross-entropy, no class weighting (classes are balanced within acceptable margins in all three datasets).

**Augmentation (training only):** Random horizontal/vertical flip (p=0.5), random rotation ±15°, color jitter (brightness=0.15, contrast=0.15, saturation=0.1).

## Evaluation

| Crop   | Classes | Test Acc | Top-2 Acc | Macro F1 | Train Epochs | Test Samples |
|--------|---------|----------|-----------|----------|-------------|--------------|
| Tomato | 10      | **99.00%** | 99.69%   | 0.986  | 15 (8 head + 7 ft) | 1,602 |
| Rice   | 10      | **96.25%** | 98.66%   | 0.960  | 25 (8 head + 17 ft) | 1,041 |
| Orange | 5       | **98.57%** | 98.57%   | 0.981  | 15 (15 head + 0 ft) | 70 |

**Per-crop performance artifacts generated:** Confusion matrix (raw + normalized), training history with fine-tune boundary, per-class precision/recall/F1 bar chart, confidence histogram, misclassified-sample grid, and class-distribution chart. All saved to `results/<crop>/`.

**Error analysis:** Tomato achieves near-perfect accuracy thanks to the largest dataset and high-quality images. Rice is slightly harder (96.25%) due to visually similar classes—`Downy Mildew` vs. `Rice Blast` account for most misclassifications, as both manifest as leaf discoloration. Orange naturally shows top-2 identical to top-1 because there are only 5 classes, but the macro F1 of 0.981 indicates the model excels even on rare classes like `Canker` and `Scab`.

**Probable failure modes:** (1) Poor lighting or blurry photos taken by budget phones, (2) early-stage infections with no visible symptoms, (3) multiple diseases on the same leaf (co-infection), (4) images of crops outside the supported set (e.g., a potato leaf submitted as Tomato).

## Software

**Backend:** FastAPI + Uvicorn server (`server.py`) serving 5 endpoints. Models are lazy-loaded on first request and stay in memory for subsequent calls. Image processing uses Pillow with PIL `Image.open()` + RGB conversion.

**Web demo:** Streamlit 1.59.2 (`app.py`) with file uploader, crop selector, animated confidence bar, per-class probability chart (matplotlib), and embedded AI chat panel. Lazy matplotlib import prevents script-thread crashes.

**Android app:** Kotlin + Jetpack Compose, Material 3, Retrofit + OkHttp for HTTP client, Coil for image loading, Android TextToSpeech API for voice playback. Navigation: single-activity with Compose destinations. Libraries: `androidx.compose.bom:2024.09.00`, `retrofit:2.9.0`, `coil:2.5.0`. Minimum SDK: 28 (Android 9). No Google Play Services dependency. The APK is 17 MB (debug build).

**AI:** `llm_advice.py` with pluggable providers—Google Gemini (package `google-genai`) and OpenCode (requests to OpenAI-compatible endpoint). Configurable via Android Settings screen. Static fallback from `treatments.json` when no API key configured.

**CI/CD:** GitHub Actions (`ci.yml`) with two jobs:
- Python: pytest (unit + e2e with random-weight models in temp dir)
- Android: Gradle test + assembleDebug (JDK 17 via actions/setup-java)

---

# Process

## Overview

The project evolved through several distinct phases:

| Phase | Effort | Outcome |
|-------|--------|---------|
| Dataset acquisition & organization | ~4h | 3 datasets structured, Orange manual download solved |
| Keras prototype → PyTorch migration | ~8h | Removed Keras/TensorFlow entirely; single-model ResNet50 |
| Two-phase fine-tuning implementation | ~6h | Per-crop models, head warmup + layer4 unfreeze, checkpointing |
| FastAPI backend + treatments.json | ~4h | 5 REST endpoints, 75 static treatment cards in 3 languages |
| Streamlit web demo | ~2h | File upload, diagnosis, treatment, AI chat |
| Android app (v1) | ~10h | Camera, gallery, predict, result display |
| LLM integration (Gemini) | ~3h | Free-tier Gemini 2.0 Flash with fallback |
| Android i18n (EN/ES/VA) | ~2h | Dictionary-based, leaf/fruit labels per crop |
| CI pipeline | ~2h | GitHub Actions for Python + Android |
| OpenCode provider + .env | ~3h | Multi-provider, qwen3.5-plus model, python-dotenv |
| Chat UX (keyboard, markdown, typing) | ~4h | imePadding, LaunchedEffect scroll, **bold** *italic*, TypingIndicator |
| Image type guardrails + tests | ~3h | JPG/PNG validation, 88 tests, local-only for real models |
| TTS + probabilities dropdown + provider title | ~4h | play/stop button, expandable list, AI name in chat |
| Kanban + documentation | ~2h | 20+ GitHub Issues, README, AGENTS.md, tech document |

**Failed approach:** Initially tried Keras + VGG16 but abandoned when torchvision ResNet50 gave better results with fewer training epochs and superior weight initialization (IMAGENET1K_V2). The entire pipeline was restructured into pure PyTorch.

## Challenges

1. **Orange dataset Cloudflare block:** Mendeley's download server blocks automated HTTP clients. Solved by documenting the manual download procedure and implementing a local-ZIP extraction path in `setup.py`.

2. **DeepSeek models returning empty content:** The OpenCode provider initially used `deepseek-v4-flash`, a reasoning model that outputs to `reasoning_content` instead of `content`. Switched to `qwen3.5-plus` which returns `content` directly in standard OpenAI format.

3. **API key persistence across terminal sessions:** Windows User environment variables weren't reliably picked up by Python subprocesses spawned from new terminals. Solved by adding `python-dotenv` support with a `src/.env` file loaded on import.

4. **Keyboard pushing chat UI:** Android's soft keyboard was hiding the chat input field. `Modifier.imePadding()` is available in Compose 1.6+ but requires `WindowInsets.ime` reference. The solution also required `LazyListState` + `LaunchedEffect` for auto-scrolling on new messages.

5. **`.gitignore` hiding Java source:** The pattern `data/` was matching the Android Kotlin data package. Fixed by changing to `/data/` (rooted path).

6. **Gradle overwriting `settings.gradle.kts`:** On some builds Gradle stripped the Kotlin plugin declaration. Required `git checkout` to restore.

7. **Test contamination from real API keys:** The `.env` file stored a valid OpenCode key, causing tests that expected API failure to instead get real LLM responses. Fixed by monkeypatching `_ask_opencode` in faillback tests and using a fake provider string (`"__test_only__"`) in e2e tests.

## Member Contribution

Single-person team. All code, including model training pipeline, backend, Android app, CI configuration, tests, treatment content, and documentation, was authored by the same developer.

---

# Conclusion

## Limitations

1. **Co-infection not handled:** The model predicts exactly one disease per leaf. If a tomato leaf has both Bacterial Spot and Early Blight, the classifier assigns probability to the most visually prominent one—potentially missing the secondary infection.

2. **Image quality sensitivity:** The model was trained on well-lit, in-focus lab and field photos. Budget smartphone cameras in low light or shaky hands significantly degrade accuracy. No low-light augmentation or denoising pipeline is in place.

3. **Crop scope:** Only 3 crops out of the 40+ major food crops worldwide. Adding a new crop requires ~1,000+ labeled images, full fine-tuning, and manual curation of treatment cards—roughly 3 working days per crop.

4. **LLM dependence on English grounding:** Even when answering in Spanish or Valencian, the AI prompt sends the static treatment in the target language but relies on the LLM's multilingual capabilities. A model with weak support for Valencian/Catalan may produce mixed-language responses.

5. **Orange dataset size (700 images):** The smallest dataset, with 98.57% accuracy, is the most vulnerable to overfitting. The per-class test set is only ~14 images.

6. **No on-device inference:** The Android app requires a server connection. Offline inference with a ~92 MB model is feasible with ONNX Runtime or TFLite but not implemented.

7. **No authentication or rate limiting:** The backend API is fully open.

## Future Improvements

1. **Multi-label classification** (sigmoid per class instead of softmax) to detect co-infections. Requires multi-label annotated data.
2. **ONNX Runtime on Android** to run the model locally, eliminating the server dependency for basic diagnosis. The ~92 MB model would be quantized to ~23 MB INT8.
3. **Active learning loop:** Allow farmers to confirm or correct diagnoses in-app; use this feedback to periodically fine-tune the model.
4. **LLM with retrieval-augmented generation (RAG):** Index a larger corpus of agricultural extension bulletins, scientific papers, and regional crop calendars. Combine with the static treatment as the primary source.
5. **Expand crop coverage** to at least Potato, Wheat, Corn, and Grape (the next most commercially significant crops).
6. **Weather integration:** Fuse leaf photos with local weather data (humidity, temperature, recent rainfall) as additional model inputs to improve disease predictions that are weather-dependent (e.g., fungal diseases).
7. **Push notifications** for disease alerts when a pest or disease is reported in the farmer's geographic area.
8. **Admin dashboard** for extension officers to monitor disease prevalence across their territory.

## Reflection

This project demonstrated that a production-quality AI application can be built from scratch using entirely free-tier infrastructure: free datasets, free pre-trained weights, free LLM APIs, and free CI minutes. The challenging part was not the AI itself—transfer learning with ResNet50 is remarkably reliable—but the integration work needed to make the AI genuinely useful to a real farmer: translating model output into actionable treatment cards, localizing them to minority languages, making them audible, and wrapping everything in a UI that works on a €100 phone.

If starting over, we would: (1) begin with multi-label classification from day one, (2) invest more heavily in data augmentation to handle real-world phone camera quality, (3) use ONNX from the start for eventual offline inference, and (4) collect user feedback earlier to validate that treatment recommendations are culturally and economically feasible for the target farmers.

The biggest lesson learned: an AI model with 99% accuracy is worthless if the farmer cannot read the results or afford the recommended treatment. The "last mile" of translation, UX, and contextualization is where real impact happens.
