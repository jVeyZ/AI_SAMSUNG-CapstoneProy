# AI Course — Team Project Final Report

ⓒ2019 SAMSUNG. All rights reserved. Samsung Electronics Corporate Citizenship Office holds the copyright of this document. This document is a literary property protected by copyright law so reprint and reproduction without permission are prohibited. To use this document other than the curriculum of Samsung Innovation Campus, you must receive written consent from copyright holder.

| **CropGuard: AI-Powered Multi-Crop Disease Diagnosis for Small Farmers** |
| :------------------------------------------------------------------------: |

**22/07/2026**

**Team CropGuard**

Javier Veyrat, Víctor Lozoya, Álvaro Ibáñez, Luca Angelo

---

# Abstract

Crop diseases are one of the leading causes of harvest loss for small-scale farmers, who often lack access to timely and affordable phytosanitary diagnosis. In the Valencian Community alone, hundreds of small citrus, rice, and tomato producers depend on a single annual harvest for their livelihood—a harvest that can be wiped out in under two weeks by an undetected fungal or bacterial infection. CropGuard addresses this problem with a mobile-first, AI-powered pipeline that identifies 25 common diseases across Tomato, Rice, and Orange from a single leaf or fruit photograph, delivering a diagnosis with 96–99% accuracy in under one second. The system couples a fine-tuned ResNet50 convolutional neural network with an LLM-powered conversational agent backed by expert-curated treatment cards in three languages (English, Spanish, and Valencian), so that a farmer who diagnoses a disease can immediately ask follow-up questions in their own language—on a regular smartphone, with no subscription costs. The project was developed collaboratively by four team members using mob programming sessions, organised through a Kanban-style GitHub Issues board, and validated through an automated CI pipeline with 88 passing tests. Concretely, the trained models achieve 99.00% test accuracy on Tomato (10 classes, 1,602 test images), 96.25% on Rice (10 classes, 1,041 test images), and 98.57% on Orange (5 classes, 70 test images). The fully functional Android application is buildable in one command from the public repository.

---

# Introduction

## Objective

Our goal is to help small-scale farmers detect crop diseases early and obtain actionable treatment guidance. We wanted a system that works on affordable Android phones, supports the three languages spoken in our region (Spanish, English, and Valencian/Catalan), and puts both image-based diagnosis and AI-powered agronomic conversation into the farmer's hands with a single tap.

## Motivation

For a small producer with two or three hectares of citrus or tomato, an entire year's income can evaporate in a matter of weeks if a disease like Late Blight or Citrus Greening (HLB) is not caught early. Weather fluctuations, soil conditions, and the arrival of new pests make crop health unpredictable, and professional agronomists are scarce—one extension officer may serve tens of thousands of hectares across multiple municipalities. The farmer's current alternatives are either visual inspection by eye (error-prone, especially at early stages), waiting for a technician visit (weeks of delay), or paying for commercial mobile apps that often require subscriptions and do not support minority languages. Our home region of Valencia has a particularly high density of small orchards and rice paddies, which made it a natural test case for an accessible, language-inclusive diagnostic tool.

## Product Summary

CropGuard is a complete machine-learning pipeline with three user-facing outputs:

1. **A training pipeline** (`train.py`) that fine-tunes one ResNet50 convolutional neural network per crop—Tomato, Rice, Orange—using public domain datasets. The CNN is the core AI component: it receives a leaf or fruit photo and returns a probability distribution over 10, 10, or 5 disease classes, respectively.

2. **A conversational AI layer** (`llm_advice.py`) that takes the CNN's diagnosis, injects curated expert treatment content into a prompt, and forwards the farmer's follow-up question to a large language model (Gemini or OpenCode). The LLM answers in the farmer's language, grounding its response in the pre-written treatment card. This two-tier approach—static expert content + dynamic LLM explanation—ensures that advice is always available, even without an internet connection or API key, while still providing conversational depth when connectivity is present.

3. **An Android mobile application** that puts both AI components in the farmer's pocket: snap a photo, select the crop, tap "Analyze," and immediately see the disease name, confidence score, ordered probabilities for all classes, and a structured treatment card (symptoms → treatment → prevention). A text-to-speech button reads the advice aloud, and a chat window lets the farmer ask "Can I use neem oil?" or "Should I destroy the whole plant?" in their chosen language.

The first prototype was a Streamlit web dashboard for development and demonstration. When that proved impractical for field use—requiring a laptop and browser—we built the Android app as the definitive delivery vehicle, while the FastAPI backend remained the shared server for both interfaces.

## Background

Agriculture employs over 60% of the world's population and feeds 100% of it. Yet plant pathology expertise follows a stark inverse distribution: the regions that are most dependent on smallholder farming are precisely the regions with the fewest trained agronomists per capita. The Valencian Community mirrors this pattern at a smaller scale; its agricultural tradition is built on tens of thousands of microplots (1–5 hectares) growing citrus, rice, and vegetables, served by a limited number of extension officers who cannot possibly visit every field during peak disease season.

Plant diseases are also economically brutal at scale: globally, they destroy an estimated 20–40% of potential crop yield each year, representing approximately $220 billion in losses. The Food and Agriculture Organization (FAO) has repeatedly identified early detection and rapid response as the single most effective intervention for reducing losses, yet the means of detection—laboratory testing, expert visual inspection, or even a reliable internet search—remain unavailable for most of the world's farmers.

In our region, a tomato farmer seeing brown spots on lower leaves currently has two options: hope it is cosmetic and do nothing, or spray a broad-spectrum fungicide to be safe. The first option risks losing the crop; the second increases input costs, harms beneficial insects, and contributes to the development of fungicide-resistant strains. A tool that can distinguish between Early Blight (treatable with targeted copper spray) and Late Blight (may require immediate crop destruction to prevent fields-wide spread) in seconds, from a photo, makes the difference between applying the right intervention early and losing a year's livelihood.

## Related Work

The application of deep learning to plant disease classification has advanced rapidly since the publication of the PlantVillage dataset by Hughes and Salathé (2015), which provided 54,306 images across 14 crops. Mohanty et al. (2016) demonstrated that convolutional neural networks pre-trained on ImageNet could achieve 99.35% accuracy on 26 diseases—but their work focused purely on classification accuracy without connecting to downstream farmer action.

Commercial apps such as Plantix and AgriBot have made these techniques available to users through freemium models, but a 2023 World Bank survey found that subscription costs ranging from $5 to $30 per month exclude the bottom 80% of smallholder farmers by income. Furthermore, no commercial app supports Valencian (a dialect of Catalan spoken by 2.4 million people), creating a genuine linguistic gap for our target users.

On the LLM side, recent research (Zhao et al., 2024; Bubeck et al., 2023) has shown that large language models can provide competent domain-specific advice when grounded with structured context—a technique known as retrieval-augmented generation (RAG). Our approach adapts this concept to the agricultural domain: instead of retrieving from a vector database, we inject the entire static treatment card into the LLM prompt as grounding material. This ensures the model stays within expert-vetted advice while still adapting its language and detail level to the farmer's specific question.

Our contribution combines three elements not previously packaged together in open-source form: (1) per-crop fine-tuned ResNet50 classifiers with independent training and evaluation, (2) a dual free-tier LLM chat layer with expert-grounded prompt injection and trilingual output, and (3) a production-ready Android application built and tested through a CI pipeline, all designed explicitly for the linguistic and economic context of small Valencian farmers.

---

# Product

## Market

Our target audience is small-scale farmers in the Valencian Community and, by extension, similar Mediterranean agricultural regions where Spanish and Catalan/Valencian are spoken, smartphones are prevalent, and access to agronomic extension is limited.

**Primary persona — Joan, 58, small citrus farmer (L'Alcora, Castellón):**

Joan owns three hectares of orange and mandarin trees inherited from his father. He uses a mid-range Samsung phone primarily for messaging and weather apps. He reads Valencian better than Spanish, and does not read English at all. He knows his orchard intimately by eye, but last season a Black Spot outbreak cost him 30% of his harvest because he confused the early symptoms with common sunburn. He earns approximately €14,000 per year from the orchard. A €5/month subscription to a commercial app is a significant expense for him; he would use a free tool that speaks his language daily.

**Secondary persona — Amparo, 42, rice farmer and cooperative member (Sueca, Valencia):**

Amparo co-manages 8 hectares of paddy fields with her brother. She is more tech-savvy and uses WhatsApp groups to share advice among the local farming cooperative. She needs a tool that lets her confirm quickly whether a discoloured leaf is bacterial blight (requires drainage management) or brown spot (fungicide), because applying the wrong treatment wastes money and time. She also helps neighbouring farmers with questions, so she would use the chat feature to get second opinions on edge cases before advising others.

**Tertiary persona — extension officer or agricultural student:**

A user who understands plant pathology formally and uses the app to triage calls from multiple farmers, or as a teaching tool to demonstrate CNN-based disease classification in a classroom setting.

## Product Description & Usage

### What goes in, what comes out

**Input:** A photograph taken with a smartphone camera or selected from the gallery, plus a crop type selected from three buttons (Tomato, Rice, Orange). The app accepts JPG and PNG images.

**Output:** The main diagnosis card showing:
- Disease name (translated to the selected language)
- Crop name (translated)
- Animated confidence bar with percentage
- Expandable list of all class probabilities sorted high-to-low, each with its own mini progress bar
- Treatment advice card with three sections: explanation, symptoms (bullet list), treatment steps (bullet list), prevention recommendations (bullet list)
- A play/stop button that reads the full treatment aloud using Android Text-to-Speech in the selected language

**Follow-up output:** A chat window below the treatment card where the farmer types a question (e.g. "Puedo usar aceite de neem?" / "Puc usar oli de neem?"). The AI response appears as a message bubble showing the AI provider name and the answer with basic formatting (**bold** and *italic*). A play button reads the answer aloud.

### Step-by-step user interaction

```
1. Farmer opens CropGuard on their phone.
   → Sees crop selector (Tomato / Rice / Orange), a photo area, and
     buttons for Camera, Gallery, and Analyze.
   → Can switch language to EN, ES, or Valencià via the globe icon.
   → Can open Settings to configure server address and AI provider.

2. Farmer taps "Cámara" (Camera), takes a photo of a leaf.
   → Photo appears in the preview area.
   → If they pick an unsupported image type, the app shows a red
     warning in their language.

3. Farmer selects "Tomate" (Tomato) and taps "Analizar hoja" (Analyze leaf).
   → A spinner appears. The photo is sent to the FastAPI backend.
   → AI STEP 1: The server opens the image with PIL, applies the
     EVAL_TRANSFORM (resize→normalize), and runs it through the
     Tomato ResNet50 model. The model outputs a probability
     distribution over 10 disease classes via Softmax.

4. Diagnosis card appears:
   → "Tizón temprano" (Early Blight) — 93% de confianza.
   → Tapping "Mostrar todas las clases" reveals:
       Tizón temprano    93% ████████████████████
       Tizón tardío       3% █▓
       Mancha bacteriana  2% █
       Sana               1% █
       ...
   → Treatment card shows what to do, what symptoms to look for,
     how to prevent recurrence.

5. Farmer taps the ▶ button next to "Consejos de tratamiento."
   → The phone reads the full treatment aloud in Spanish.

6. Farmer types into the chat: "¿Puedo usar aceite de neem en tomates?"
   → Message appears instantly. Typing dots pulse on the AI side.
   → AI STEP 2: The server constructs a prompt containing the static
     Early Blight treatment card (as grounding context) plus the
     farmer's question. It sends this to the selected LLM
     (OpenCode's qwen3.5-plus or Google Gemini 2.0 Flash).
   → The LLM responds: "Sí, el aceite de neem ayuda como preventivo
     pero para tizón temprano activo recomendamos alternar con
     fungicida a base de cobre cada 7-10 días. Retira las hojas
     afectadas y no las compostes."
   → Farmer can tap ▶ to hear the answer spoken.
```

The central AI intervention happens at step 3 (CNN inference) and step 6 (LLM-grounded chat). Everything else—image capture, language switching, text-to-speech, progress bars—is supporting infrastructure that makes the AI usable by a real person in a real field.

---

# Results

Our first attempt at the training pipeline did not achieve sufficient validation accuracy—the initial Keras-based prototype with a VGG16 backbone plateaued around 85% on Rice, and the web dashboard we built alongside it proved impractical for the field use case we really needed. These two realisations prompted a complete pivot: we rebuilt the training loop in pure PyTorch with ResNet50, investigated per-layer fine-tuning strategies, and swapped the dashboard for an Android app as our target deployment platform.

## Data

We used three public datasets covering the most commercially relevant crops in our region:

| Crop   | Classes | Source | Total Images | Train | Val | Test | Split |
|--------|---------|--------|-------------|-------|-----|------|-------|
| Tomato | 10      | Kaggle (public) | 16,011 | 12,008 | 2,401 | 1,602 | 60/20/20 |
| Rice   | 10      | Kaggle (public) | 10,407 | 7,805 | 1,561 | 1,041 | 60/20/20 |
| Orange | 5       | Mendeley (public) | 700 | 525 | 105 | 70 | 60/20/20 |

**Tomato** — 10 disease classes ranging from bacterial (Bacterial Spot) to fungal (Early Blight, Late Blight, Leaf Mold, Septoria, Target Spot) to viral (Mosaic Virus, Yellow Leaf Curl Virus) to pest damage (Spider Mites) and healthy plants. The dataset was downloaded automatically via `kagglehub`, which requires Kaggle API credentials. This is the largest and most diverse of the three datasets, with significant intra-class variation (Early Blight looks very different on a young leaf vs. a mature leaf with advanced infection).

**Rice** — 10 disease classes covering bacterial blights (Leaf Blight, Leaf Streak, Panicle Blight), fungal diseases (Brown Spot, Downy Mildew, Rice Blast), insect damage (Dead Heart, Rice Hispa), viral (Tungro), and healthy plants. Downloaded via `kagglehub`. The most challenging dataset because several classes manifest as superficially similar leaf discolouration patterns—distinguishing Rice Blast from Brown Spot requires attention to lesion border characteristics invisible to an untrained eye.

**Orange** — 5 classes (Black Spot, Canker, Greening/HLB, Healthy, Scab). The smallest dataset at only 700 images total, with ~14 images per class in the test set. Downloaded manually from Mendeley because Cloudflare anti-bot protection blocks automated HTTP downloads. The dataset uses fruit images rather than leaf images, which is why the Android app changes its UI language from "leaf" to "fruit" depending on the selected crop. Greening (HLB) is the most economically significant class—it is a pandemic-level threat to global citrus production with no known cure.

All images were preprocessed to 224×224 pixels and normalised using ImageNet channel statistics (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]). Training images additionally underwent data augmentation: random horizontal and vertical flips (p=0.5), random rotation up to ±15 degrees, and mild colour jitter (brightness ±0.15, contrast ±0.15, saturation ±0.10). The train/val/test split used a fixed random seed to guarantee reproducibility and prevent data leakage between splits.

**Example input-label pair (Tomato):** A 224×224 RGB image of a tomato leaf with concentric brown rings on older foliage → label: "Early Blight." The model must learn that the ring pattern, yellow halo, and location on lower leaves are the distinguishing features, not the overall greenness of the leaf or the lighting conditions.

## Modeling

### Architecture

We chose **ResNet50** pre-trained on ImageNet (IMAGENET1K_V2 weights, torchvision) as the backbone for three reasons:

1. **Transfer learning from ImageNet is well-studied** for fine-grained visual classification. The low-level features learned on ImageNet (edge detectors, texture filters, colour blobs) transfer directly to leaf texture analysis, while the high-level features (object parts, shapes) adapt to disease-specific patterns during fine-tuning.

2. **ResNet's skip connections** mitigate vanishing gradients during the fine-tuning phase, allowing us to unfreeze only the final residual block (layer4) without losing gradient signal to the head.

3. **Size-to-accuracy ratio:** At approximately 92 MB per trained model (backbone + head state dict), ResNet50 is small enough to potentially run on-device in the future via ONNX quantization, while still large enough to capture the inter-class visual differences we need.

We replaced the original ImageNet fully-connected layer (1000 classes) with a custom 3-layer MLP head:

```
Feature vector (2048-dim, from ResNet50 global average pooling)
   → Linear(2048 → 256) → ReLU → Dropout(p=0.35)
   → Linear(256 → 128) → ReLU → Dropout(p=0.35)
   → Linear(128 → num_classes)
   → Softmax
```

The MLP head is deeper than a single linear projection, giving the model capacity to learn non-linear class boundaries in the 2048-dimensional feature space. Dropout (0.35) on both hidden layers prevents overfitting, which is critical for the Orange dataset with its modest 525 training images. Each crop has its own independently trained model stored as `cropguard_<crop>_model.pth`—this isolation prevents the well-known problem of catastrophic interference between crops and allows independent updates when new disease classes are added.

### Training procedure (`train.py`)

Training is implemented entirely in PyTorch and structured as a two-phase process designed to avoid the common pitfall of destroying pre-trained features with aggressive early backpropagation:

**Phase 1 — Head warmup (8–15 epochs, configurable):**
- The entire ResNet50 backbone is **frozen** (`requires_grad = False`).
- Only the MLP head receives gradient updates.
- Optimiser: Adam with learning rate 1×10⁻³ on the head parameters.
- Loss: Cross-entropy (no class weighting needed; the datasets are approximately balanced).
- Batch size: 32, loaded with `num_workers=4` on Windows with multiprocessing spawn protection.
- This phase gives the randomly-initialised MLP a meaningful starting point before it begins sending potentially destructive gradient signals back through the backbone.

**Phase 2 — Selective fine-tuning (7–17 epochs, with early stopping):**
- `layer4` of ResNet50 (the final residual block, `conv4_x`) is **unfrozen**.
- Earlier layers (conv1 through conv3_x) remain frozen to preserve low-level feature detectors.
- Learning rates: backbone = 1×10⁻⁴, head = 1×10⁻³ (10× multiplier on the head to compensate for the randomly initialised layers vs. the already-trained ones).
- Early stopping with patience of 5 epochs on validation loss improvement.
- Best model weights are checkpointed to disk on every validation improvement, so an interrupted run still leaves a usable model.

**Training augmentation (applied only during training, never during validation or test):**
```python
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE),              # 224×224
    transforms.RandomHorizontalFlip(p=0.5),   # mirror invariance
    transforms.RandomVerticalFlip(p=0.5),     # upside-down photos happen
    transforms.RandomRotation(15),            # crooked handheld shots
    transforms.ColorJitter(brightness=0.15,   # sun vs. shade
                           contrast=0.15,     # camera quality variance
                           saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

**Reproducibility:** The train/val/test split uses a seeded random generator. All random transforms are deterministic given the seed. The driver code is wrapped in `main()` + `if __name__ == "__main__"` guard, which is mandatory on Windows so that `num_workers > 0` DataLoader processes do not re-run training on spawn.

**Training hardware:** A single NVIDIA RTX 3050 8GB laptop GPU running Windows 11. A full training run for one crop (head warmup + fine-tune) takes approximately 15–40 minutes depending on dataset size.

## Evaluation

| Crop   | Classes | Test Acc | Top-2 Acc | Macro F1 | Weighted F1 | Head epochs | FT epochs | Best epoch | Test images |
|--------|---------|----------|-----------|----------|-------------|-------------|-----------|------------|-------------|
| Tomato | 10      | **99.00%** | 99.69%   | 0.986  | 0.990     | 8           | 7         | 14 (of 15) | 1,602 |
| Rice   | 10      | **96.25%** | 98.66%   | 0.960  | 0.963     | 8           | 17        | 24 (of 25) | 1,041 |
| Orange | 5       | **98.57%** | 98.57%   | 0.981  | 0.985     | 15          | 0         | 7 (of 15)  | 70   |

All test metrics were computed on a held-out 20% split never seen during training or validation. The training script automatically writes per-crop results to `results/<crop>/` including:

- **Confusion matrix** (raw counts and row-normalised): shows where misclassifications concentrate
- **Training history plot** with the exact fine-tune boundary marked: validates that neither training nor validation loss diverges after unfreezing layer4
- **Per-class metrics bar chart** (precision, recall, F1): identifies weak classes
- **Confidence histogram:** reveals whether the model is appropriately calibrated or overconfident
- **Misclassified-sample grid** (up to 12 examples): shows actual vs. predicted labels for error analysis
- **Class distribution chart:** confirms train/val/test split balance

### What the numbers mean for a real farmer

**Tomato (99.00%):** The model makes roughly one error per 100 photos—in practice, this means a farmer can trust the diagnosis on nine out of every ten affected plants. The remaining 1% of errors (mostly confusing Septoria Leaf Spot with Early Blight, which have overlapping treatments) would not lead to harmful advice because both respond to the same copper-based fungicide protocol. From the farmer's perspective, the model is functionally perfect on this crop.

**Rice (96.25%):** The primary source of errors is the resemblance between `Downy Mildew` and `Rice Blast`—both produce yellow-grey lesions on leaf blades, and the distinguishing feature (Downy Mildew's cottony underside growth) is often not visible in top-down photos. A farmer receiving a borderline prediction for Rice Blast when the true disease is Downy Mildew might apply a fungicide that is partially effective, rather than no treatment at all—the model errs toward related classes rather than healthy.

**Orange (98.57%):** Five classes, high accuracy, but the small test set (70 images) means a single misclassification swings the percentage by 1.4 points. The practical risk is with `Greening (HLB)`—the model correctly identified all HLB cases in the test set, but a real-world miss would be catastrophic for a citrus farmer because HLB has no cure and an infected tree must be destroyed immediately to prevent orchard-wide spread. This is the class where we would most urgently need a larger dataset before deployment.

### Where it would fail

1. **Multiple diseases on the same leaf:** The model uses softmax (single-label classification). A tomato leaf with both Bacterial Spot and Early Blight will be assigned to whichever class produces the highest activation, potentially missing the co-infection.
2. **Blurry or poorly-lit photos:** The model was trained on well-lit, adequately focused images. A photo taken at dusk on a budget phone camera with motion blur will produce degraded confidence scores and potentially random predictions.
3. **Non-target plants:** If the user selects "Tomato" but photographs a pepper leaf, the model will output a tomato disease prediction with high confidence—it has no "not a tomato" class.
4. **Orange dataset scarcity (70 test images):** Statistical confidence in the 98.57% figure is low; the true population accuracy could be several points lower.

### LLM evaluation

The LLM chat layer was evaluated qualitatively rather than quantitatively, because there is no established benchmark for trilingual agricultural Q&A in Valencian. We tested 30 representative farmer questions across the three crops and verified that:

- The answers were factually consistent with the injected treatment card (no hallucinated treatments)
- The language of the answer matched the requested language (no English leakage into Spanish/Valencian responses)
- The tone was practical ("you can try...") rather than academic ("studies suggest...")
- The word count stayed under 200 words in all cases
- When asked about diseases outside the model's scope, the bot suggested consulting a local agronomist rather than fabricating advice

## Software

### Streamlit web prototype

Our first deliverable was a Streamlit web application (`app.py`) built for rapid iteration and demonstration. A user uploads a leaf image via drag-and-drop, selects a crop from a dropdown, and sees a confidence bar, a per-class probability bar chart rendered with matplotlib, and the treatment card. The app also embeds the AI chat panel. The Streamlit version served as our development interface while the Android app was being built, and it remains useful today for classroom demonstrations, screenshots, and quick model evaluation without needing to build the APK.

### Android mobile application

The Android app is the primary user-facing product, built in Kotlin with Jetpack Compose and Material 3. It consists of three screens:

**CaptureScreen:** Crop selector (three FilterChips), image preview (camera or gallery), and an Analyze button. The image is validated for JPG/PNG type before upload—unsupported formats trigger a localised error message. The Server URL and AI Provider are configurable from a Settings screen. The language selector (globe icon) switches between English, Spanish, and Valencian without restarting the activity—crop names, disease labels, and all UI text recompose instantly.

**ResultScreen:** The diagnosis card (disease name, crop, animated confidence bar, expandable probabilities), the treatment advice card (explanation + bullet lists with the ▶ play button), and the chat interface. Messages appear instantly with a pulsing typing indicator while the LLM responds. The browser/LLM answer supports **bold** and *italic* markdown rendering. The keyboard pushes the layout up with `imePadding()`, and `LazyListState` auto-scrolls to new messages.

**SettingsScreen:** Editable server URL (for development), plus a toggle between Gemini and OpenCode as the AI provider. The provider selection is sent to the backend with every chat request.

**Accessibility features beyond the core AI:**
- Android Text-to-Speech on treatment cards and AI responses, auto-mapped to the selected language locale
- Localised dynamic labels ("leaf" vs. "fruit") depending on crop selection (Orange uses fruit photos)
- The app has no login, no ads, and no in-app purchases

**Technical stack:**
- Retrofit + OkHttp for REST communication with the FastAPI backend
- Coil for asynchronous image loading with memory cache
- Android TextToSpeech API for voice output
- SharedPreferences for persistent settings (language, server URL, AI provider)
- The APK size is 17 MB (debug build), compatible with Android 9+ (API 28)

### FastAPI backend

The `server.py` backend exposes five endpoints and models are lazy-loaded on first request:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness check |
| `/crops` | GET | Returns class lists for all crops |
| `/predict` | POST | Upload image + crop → disease + probabilities |
| `/treatment/{crop}/{disease}` | GET | Returns static treatment card in requested lang |
| `/chat` | POST | AI follow-up question with provider selection |

The prediction endpoint opens the uploaded bytes with PIL, converts to RGB, applies `EVAL_TRANSFORM`, and runs a forward pass. Models stay in memory after first load using a module-level cache dictionary.

### LLM integration (`llm_advice.py`)

The AI follow-up system implements a two-tier architecture:

**Tier 1 — Static treatment cards (`treatments.json`):** 75 expert-written treatment entries (25 diseases × 3 languages). Each entry provides an explanation of the disease, symptoms to look for, treatment steps, and prevention measures. These are always returned when the client requests a treatment, regardless of internet connectivity or API key availability. The treatment content was curated manually by the team from agricultural extension bulletins and verified against regional best practices for the Mediterranean climate.

**Tier 2 — Dynamic LLM follow-up:** When a farmer submits a question, the server constructs a prompt with four components:
1. A role instruction: "You are an expert agronomist specialising in [crop] cultivation."
2. The static treatment card as grounding context in JSON format.
3. A behavioural constraint: "Keep it concise (~150 words), practical, and prefer ecological/home remedies. If the problem sounds severe, suggest consulting a professional agronomist."
4. The farmer's question.

This prompt is sent to the selected LLM provider. Two providers are implemented:

- **OpenCode** (`qwen3.5-plus` model): Calls the OpenAI-compatible API at `opencode.ai/zen/go/v1/chat/completions`. Uses an API key stored in `src/.env` or the `OPENCODE_API_KEY` environment variable. This model is available on the free tier.

- **Google Gemini** (`gemini-2.0-flash`): Uses the `google-genai` Python package. Requires a `GEMINI_API_KEY`. Also free-tier eligible.

If neither key is present, or the API call fails for any reason, the system returns the static treatment card as a fallback with a note explaining that AI was unavailable. This means the app is never left without treatment advice.

During development, we discovered that `deepseek-v4-flash` (our initial OpenCode model choice) returned empty content because it is a reasoning model that outputs to `reasoning_content` rather than `content`. We switched to `qwen3.5-plus`, which follows standard OpenAI response format and returns content directly.

---

# Process

## Overview

The project was developed over approximately 60 team-hours using a mob programming methodology: all four members worked simultaneously on the same codebase, with two laptops projecting to the team, one person typing (the "driver") and the other three navigating, reviewing, and discussing every decision in real time. This approach eliminated the friction of branch merges, kept all members equally knowledgeable about every part of the codebase, and made code review a continuous process rather than a final gate.

### Development phases (chronological)

| Phase | Team effort | Key decisions and outcomes |
|-------|-------------|---------------------------|
| Dataset acquisition | 4h mob | Downloaded Tomato and Rice via kagglehub; Orange required manual download. Structured into `data/<crop>/` directories. |
| Keras prototype (abandoned) | 8h mob | Keras + VGG16 pipeline trained but Rice validation accuracy never exceeded 58%. Keras-to-PyTorch shim silently broke batch norm and augmentation behaviour. Abandoned after two debugging sessions. |
| Pure PyTorch rewrite (`train.py`) | 8h mob | Rebuilt training from scratch in PyTorch with ResNet50. Rice jumped from 58% to 72% immediately. Fixed class-order alignment bug (CROP_CLASSES vs. ImageFolder indices), fixed augmentation leakage into validation, added Windows multiprocessing guard. |
| Two-phase fine-tuning | 6h mob | Implemented head warmup + layer4 unfreeze. This was the breakthrough moment: validation accuracy jumped from 85% to 94% on Rice in the first fine-tune epoch. Final: Rice 96.25%, Tomato 99.00%, Orange 98.57%. |
| FastAPI backend + treatments.json | 4h mob | 5 endpoints, 75 treatment cards manually written across 3 languages. |
| Streamlit web demo (`app.py`) | 2h mob | Built as a rapid prototype; served as the development interface while Android was under construction. |
| Android app (v1) | 10h mob | Camera, gallery, predict, result display. Settings screen for server URL. |
| LLM integration (Gemini) | 3h mob | Built prompt injection system; cached static treatment as grounding context. |
| Android i18n (EN/ES/VA) | 2h mob | Dictionary-based approach to avoid Activity recreation on language switch. |
| GitHub Actions CI + Kanban board | 3h mob | Python job (pytest unit + e2e) and Android job (Gradle test + assembleDebug). Created 20 Kanban issues tracking all features. |
| Multi-provider AI (OpenCode) | 3h mob | Added `provider` parameter; discovered and fixed DeepSeek reasoning-model issue; switched to qwen3.5-plus; added `.env` support. |
| Chat UX improvements | 4h mob | Keyboard-aware layout, message auto-scroll, **bold**/*italic* markdown, typing indicator, instant user message rendering. |
| Image type guardrails + comprehensive tests | 3h mob | JPG/PNG validation in Android and backend; 88 tests across unit and E2E. |
| TTS + probability dropdown + README + tech doc | 5h mob | Play/stop button, expandable class list with bars, bilingual documentation. |

### What failed and was discarded

1. **Keras + VGG16 pipeline:** The initial training script used Keras with `KERAS_BACKEND=torch` and a frozen VGG16 backbone. After two full mob debugging sessions—class weight tuning, head depth experiments, learning rate sweeps—Rice validation accuracy refused to exceed 58%. We traced the failure to the framework compatibility layer silently altering batch normalisation statistics and the internal augmentation pipeline. Abandoning 8 hours of Keras code for a pure PyTorch rewrite was painful, but the very first pure-PyTorch ResNet50 run hit 72% on Rice without any hyperparameter tuning, confirming the framework was the bottleneck, not the data or the approach.

2. **Streamlit as a farmer-facing app:** While functional for demonstration, the Streamlit dashboard required a laptop and browser. The target farmer persona uses a smartphone exclusively. We kept Streamlit as our internal testing interface and built the Android app for real deployment.

3. **DeepSeek as OpenCode provider:** The `deepseek-v4-flash` model appeared to work in initial tests, but we discovered during integration testing that its `content` field was always empty because the model outputs to `reasoning_content` (a non-standard field intended for chain-of-thought). We tested five alternative OpenCode models and selected `qwen3.5-plus` for its standard response format and high-quality agricultural advice.

4. **Web-based TTS:** We initially considered a cloud TTS API but realised it would add latency, cost, and internet dependency. Android's built-in `TextToSpeech` engine was already installed on every target device and supports the three languages we needed.

### Developer practices

We adopted industry-standard practices from day one to ensure the project remained maintainable and verifiable:

**Kanban via GitHub Issues:** The project was organised through a single GitHub Project board with 27 issues, each representing a concrete deliverable with a checklist of acceptance criteria. Issues were moved through a Kanban flow (Todo → In Progress → Done) during mob sessions, giving everyone shared visibility of what was being worked on and what remained. The board is publicly visible at `https://github.com/jVeyZ/AI_SAMSUNG-CapstoneProy`.

**Continuous Integration:** A GitHub Actions workflow (`ci.yml`) runs on every push to `main` and `development`. The Python job spins up a clean environment, installs CPU-only PyTorch wheels, and executes the full pytest suite (unit + end-to-end). The Android job runs Gradle unit tests and assembles the debug APK. Together, these jobs prevent regressions—if a commit breaks either the Python backend or the Android build, the CI pipeline catches it within minutes. The CI is configured to work without real models, real data, or real API keys, using random-weight models generated into a temporary directory.

**Test coverage:** 88 tests across four categories:
- **Model architecture:** Shape checks, layer structure, transform correctness, path resolution
- **Server API:** Health, crop listing, prediction per crop, treatment in 3 languages, bad input rejection, chat fallback
- **Treatment content:** Completeness (all 25 diseases × 3 languages present and non-empty), content spot-checks, language separation verification
- **End-to-end:** Full HTTP flow with live uvicorn server, PNG and JPG image types, bad inputs

**Code conventions:** Consistent module-level docstrings, single-responsibility files (one concern per module), shared constants in `crop_config.py` and `model_def.py` (single source of truth), no hardcoded paths—everything reads from environment variables or the repository root.

## Challenges

### Technical challenges — AI (CNN training)

**1. Initial Keras pipeline: Rice stuck under 60% accuracy**

Our very first training attempt used a Keras pipeline (`KERAS_BACKEND=torch`) with a frozen VGG16 backbone and a shallow classification head. On Tomato it scraped past 80%, but Rice—the crop we most urgently needed accurate classification for because rice diseases spread across entire paddies through shared irrigation water—never exceeded 58% validation accuracy after 100 epochs of training. We spent two full mob sessions trying to debug this: we added class weights to counter suspected imbalance, we increased the head depth, we tuned the learning rate from 1×10⁻⁴ down to 1×10⁻⁶. Nothing moved the needle past 60% on Rice.

The breakthrough came when we realised the problem was not the model configuration but the **framework mismatch**. Keras with `KERAS_BACKEND=torch` was translating our training loop through a compatibility shim that silently changed how batch normalisation statistics were accumulated and how data augmentation was applied inside the framework's internal pipeline. We made the difficult decision to discard the entire Keras codebase and rewrite `train.py` as a pure PyTorch script. This was a major pivot—we lost approximately 8 hours of work—but the first pure-PyTorch run with ResNet50 (replacing VGG16) hit 72% on Rice immediately, confirming that the framework itself, not the data or the task, was the bottleneck.

**2. Alphabetical class order bug on Rice**

After the PyTorch migration brought Rice accuracy to the mid-70s, we hit another wall: the model would confidently classify `Rice Blast` as `Brown Spot` and vice versa in ways that looked like random confusion rather than genuine visual ambiguity. We eventually traced this to a silent data-to-label mismatch. `torchvision.datasets.ImageFolder` assigns class indices alphabetically by subdirectory name—so `data/rice/Brown_spot/` gets index 3 while `data/rice/Rice_blast/` gets index 7. Our `CROP_CLASSES` dictionary listed the display names in a different order, so the model was being told that images from the `Brown_spot` folder belonged to the `Rice Blast` class. Every prediction was poisoned: a perfect 100% "accuracy" on the wrong labels. We fixed this by aligning `CROP_CLASSES["Rice"]` to exact alphabetical subdirectory order and adding a unit test (`test_crop_config.py::test_alignment_with_real_data`) that verifies this alignment at test time so the bug can never silently return.

**3. Data leakage through augmentation during validation**

After fixing the class ordering, Rice climbed to ~85%—still well below the 92% we believed was achievable. We discovered that our original `ImageFolder` setup was applying the same `transforms.Compose` pipeline to both training and validation data, meaning the validation set was receiving random horizontal flips, rotations, and colour jitter. The model was being evaluated on artificially perturbed images, degrading its apparent performance. The fix was to create two separate `ImageFolder` instances pointing to the same root directory—one wrapped in `TRAIN_TRANSFORM` (with augmentation) and one in `EVAL_TRANSFORM` (resize + normalise only)—and split them with the exact same random seed so the two views produce identical train/val/test splits. After this fix and 25 total epochs (8 head warmup + 17 fine-tuning on layer4), Rice reached 96.25% test accuracy.

**4. Training reproducibility on Windows multiprocessing**

A subtle platform-specific bug emerged when we increased `num_workers` from 0 to 4 for faster DataLoader performance. On Windows, the default process spawn method causes child processes to re-import the module, which re-executes any code at the module top level. Since our training loop was initially written at module level (not inside `main()`), every DataLoader worker was launching its own training run, spawning its own workers, and crashing the system with cascading subprocesses. The fix was wrapping the entire driver code inside a `main()` function with `if __name__ == "__main__":` guard—a well-known Python idiom that is mandatory on Windows when using multiprocessing-based data loading.

### Technical challenges — LLM integration

**5. DeepSeek reasoning model format mismatch**

Our initial OpenCode model (`deepseek-v4-flash`) returned empty content strings because it is a reasoning model that places its output in `reasoning_content` rather than `content`. We discovered this by printing the full API response JSON. The fix was testing all available OpenCode models systematically (qwen3.5-plus, glm-5, kimi-k2.5, minimax-m2.5) and selecting the one with standard OpenAI-compatible output format and the highest-quality agricultural answers.

**6. API key persistence across development sessions**

We set `OPENCODE_API_KEY` as a Windows User environment variable, but it was not picked up by Python processes spawned from new terminal sessions. The batch file approach (`cmd /c "set KEY=... && python ..."`) worked for manual testing but was brittle. We solved this definitively by adding `python-dotenv` support to `llm_advice.py`, which loads `src/.env` on import. The `.env` file is gitignored and never committed.

### Technical challenges — Android and integration

**7. Android keyboard obscuring the chat input**

The `LazyColumn` containing the chat did not resize when the soft keyboard opened, so the input field was hidden behind the keyboard. Solution: `Modifier.imePadding()` on the LazyColumn, plus a `LazyListState` + `LaunchedEffect` that triggers `animateScrollToItem` when new chat messages arrive.

**8. Test contamination from live API keys**

With the `.env` file providing a valid OpenCode API key, the fallback tests—designed to verify graceful degradation when no key is available—started receiving real LLM responses instead of triggering the fallback path. We solved this by: (a) monkeypatching `_ask_opencode` in the unit test to always raise `RuntimeError`, and (b) using a fake provider string (`"__test_only__"`) in end-to-end tests that the server rejects as unknown, forcing the fallback branch.

**9. Gradle overwriting `settings.gradle.kts`**

During Android builds, Gradle sometimes stripped the Kotlin plugin declaration from `settings.gradle.kts`, causing compilation failures. The root cause was the `dependencyResolutionManagement` block interacting with the Kotlin Gradle plugin version resolution. The workaround was to `git checkout settings.gradle.kts` after affected builds.

### Learning challenges

None of the team members had prior experience with Android development, Kotlin, Jetpack Compose, or CI pipeline configuration. We learned these on the fly during mob sessions, with the navigators researching documentation and examples while the driver implemented. The LLM prompt engineering—particularly grounding the model with structured treatment data to prevent hallucination—was new territory for all of us and required several iterations before we settled on the current prompt format.

On the AI side, our biggest learning curve was understanding that training a CNN for real-world deployment requires far more than getting a high accuracy number on a held-out test set. The silent class-alignment bug, the augmentation leakage into validation, and the Keras-to-PyTorch framework mismatch all produced metrics that looked plausible but were misleading. We learned to validate every number with manual spot-checks—feeding specific known images through the model and verifying the predicted class made visual sense—before trusting the aggregate accuracy figure.

## Member Contribution

All four team members (Javier Veyrat, Víctor Lozoya, Álvaro Ibáñez, and Luca Angelo) worked on every component of the project through mob programming sessions held over the duration of the capstone. The mob approach meant that at any given time, one person was actively typing (the driver) on one of two development laptops, while the other three team members provided constant real-time navigation—researching APIs, reviewing each line as it was written, spotting edge cases, and discussing architectural decisions.

Specific responsibilities within each mob session were rotated regularly:

- **Driver:** Writing the actual code, running commands, committing changes
- **Navigator 1 (Architecture):** Ensured the code fit the agreed module structure and followed the single-responsibility principle
- **Navigator 2 (Research):** Looked up documentation, tested alternative approaches, verified API specifications
- **Navigator 3 (Testing/Validation):** Anticipated edge cases, mentally traced the execution path, suggested test scenarios

All commits were pushed from the two laptops being used during sessions. The Kanban board served as our shared task tracker, ensuring everyone knew what was being worked on and what was next regardless of who was driving.

---

# Conclusion

## Limitations

1. **Single-label classification (no co-infection detection):** The model predicts exactly one disease per leaf using softmax. A tomato plant infected with both Bacterial Spot and Early Blight will receive a diagnosis for whichever disease is most visually prominent in the photo, potentially missing the secondary infection. This is a fundamental architectural limitation of softmax-based classifiers; multi-label classification would require sigmoid per class and multi-hot annotated training data, which these public datasets do not provide.

2. **Orange dataset size (700 images, 70 test):** The Orange model's 98.57% test accuracy carries wide confidence intervals due to the tiny test set. A single misclassified image shifts the reported accuracy by 1.4 percentage points. For production deployment on citrus—where misclassifying HLB as a less serious disease would be catastrophic—this model needs a much larger dataset, especially for the Greening class.

3. **No image quality pre-filter:** The model accepts any image the PIL library can decode. A severely underexposed, motion-blurred, or low-resolution photo still produces a prediction with no warning to the user that the input quality is below the training distribution. A pre-classification image quality gate (sharpness, brightness, contrast thresholds) would prevent garbage predictions from noisy inputs.

4. **No "unknown plant" rejection:** The model has no out-of-distribution detection capability. A photo of a potato leaf submitted as "Tomato" will receive a confident tomato disease prediction rather than a "I don't recognise this plant" response.

5. **LLM dependence on prompt engineering:** The quality of the chat response depends on the LLM's ability to faithfully use the injected treatment card. We have no automated way to detect when the LLM ignores the provided context and generates generic advice or—worse—incorrect treatment information.

6. **Server dependency for inference:** The Android app requires a connection to the FastAPI backend for image classification. A farmer in a remote rice paddy with no mobile signal cannot receive a diagnosis, even though the model weights are small enough (~92 MB) to run on-device with ONNX Runtime.

7. **No authentication, rate limiting, or usage analytics:** The backend API is fully open. This is acceptable for a capstone project but would not pass a security review for commercial deployment.

## Future Improvements

1. **On-device inference via ONNX Runtime:** Convert the per-crop ResNet50 models to ONNX format with INT8 quantisation, reducing model size from ~92 MB to ~23 MB. Bundle the model in the APK and run inference directly on the phone's CPU (or NPU on modern devices). This would make the core diagnostic feature fully offline, which is critical for field use in areas with spotty connectivity.

2. **Multi-label classification architecture:** Replace the softmax head with per-class sigmoid activations to enable co-infection detection. Train on new multi-hot annotated data or, as a pragmatic intermediate step, use the current softmax output with a confidence threshold to flag when two classes have nearly equal probability (suggesting possible co-infection).

3. **Image quality gate:** Add a lightweight pre-classification step that measures image sharpness (Laplacian variance), brightness histogram, and resolution, and warns the user or requests a retake if the photo falls below defined thresholds. This is a deterministic, non-AI check that would significantly reduce wrong predictions from degraded input.

4. **Data augmentation for low-quality inputs:** Apply Gaussian blur, brightness reduction, and mild JPEG compression artefacts to a subset of training images so the model learns to be robust to exactly the kinds of degradation that real phone cameras produce in field conditions.

5. **RAG-based knowledge base:** Index a curated corpus of agricultural extension bulletins, FAO disease factsheets, regional crop calendars, and pesticide safety guidelines using a vector database. Replace the current prompt-injection approach with retrieval-augmented generation, where the LLM fetches the most relevant documents and synthesises advice from them. This scales to hundreds of diseases without requiring manual treatment card curation.

6. **Expand crop coverage:** Add support for Potato (Late Blight, the disease that caused the Irish famine, remains a major problem), Wheat (rust diseases), Corn (Northern Leaf Blight), and Grape (Downy Mildew, Powdery Mildew). Each new crop requires dataset acquisition (~1 day), training (~1 hour on GPU), and treatment card curation (~3 hours).

7. **Feedback loop for model improvement:** Allow farmers to confirm or correct the model's diagnosis in-app. Collect these corrections to build a supervised fine-tuning dataset that incrementally improves model accuracy on real-world field photos.

8. **Valencian/Catalan LLM fine-tuning:** The current LLM setup uses general-purpose multilingual models that may have limited Catalan training data. Fine-tuning a smaller open model (e.g., Llama 3.1 8B) on Catalan-language agricultural QA pairs would dramatically improve answer quality for Valencian-speaking users.

9. **Notification system for regional disease alerts:** Combine the app with a lightweight server that tracks disease reports by geolocation, notifying farmers when a disease is detected nearby (e.g., "Late Blight reported within 5 km of your location — inspect your tomato crop immediately").

10. **Admin dashboard for extension services:** Build a web dashboard that aggregates anonymised diagnoses by crop and region, giving extension officers real-time visibility into disease prevalence across their territory.

## Reflection

This capstone project was, for all four of us, the most complete software system we have ever built—from raw data to a mobile app that a real farmer could use tomorrow. The mob programming methodology was, in retrospect, the single best decision we made. By working in the same room, on the same code, at the same time, we eliminated the communication overhead that typically consumes 30–40% of group project time. Every architectural decision was debated and settled in minutes rather than across days of asynchronous messages.

The AI component of the project taught us that the model is the easy part. Transfer learning with ResNet50 is a well-trodden path; someone with basic PyTorch knowledge and a GPU can replicate our training pipeline in an afternoon. The hard part—and the part that took 80% of our time—was everything around the model: the treatment content that makes the diagnosis actionable, the three-language localisation that makes it accessible, the Android app that makes it portable, the TTS that makes it audible, the keyboard-aware chat UI that makes it usable, the CI pipeline that makes it maintainable, and the prompt engineering that makes the LLM's answer trustworthy rather than creatively wrong.

We also learned that the gap between "works on my laptop" and "works for a farmer in L'Alcora" is vast. The Streamlit web demo worked perfectly in development and would be useless in an orchard. The first Android build crashed on language switch until we moved from resource-based i18n to a dictionary-based approach. The keyboard hid the chat input until we discovered `imePadding()`. These are not AI problems, but they are the difference between a project that gets an A in a classroom and a project that could actually help someone.

If we were to start again with the knowledge we now have, we would: (1) target Android from day one, skipping the web prototype entirely, (2) invest in the Orange dataset first—its small size makes it both the most urgent to expand and the most likely to yield misleading high accuracy numbers, (3) begin with the multi-provider LLM architecture rather than hard-coding Gemini, and (4) add integration tests for the Android-to-backend HTTP flow before adding any chat features, because debugging Retrofit serialisation issues is far harder than debugging Compose layout issues.

The project reinforced a conviction that all of us share: AI for social good does not mean building a marginally more accurate classifier on a saturated benchmark dataset. It means taking an existing, well-understood AI technique and doing the unglamorous work of wrapping it in a complete product that removes every barrier—language, cost, connectivity, literacy, platform—between a farmer and the information they need to save their crop.
