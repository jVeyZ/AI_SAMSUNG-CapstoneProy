"""
CropGuard - Streamlit Interactive Demo
Multi-Crop Disease Diagnosis (Tomato, Rice, Orange)
"""
import os, sys, io

import streamlit as st
import numpy as np
import torch
from PIL import Image
from crop_config import CROP_CLASSES, get_disease_name, get_num_classes
from model_def import build_model, model_path, EVAL_TRANSFORM

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- LLM / TTS (graceful degradation if packages missing) ----------------
_groq_available = True
try:
    from openai import OpenAI
except ImportError:
    _groq_available = False

_gtts_available = True
try:
    from gtts import gTTS
except ImportError:
    _gtts_available = False


def obtener_tratamiento_llm(enfermedad, cultivo):
    if not _groq_available:
        return "[LLM no disponible — instala `openai` y configura `st.secrets.GROQ_API_KEY`]"
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets.get("GROQ_API_KEY", ""),
    )
    respuesta = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "system",
            "content": (
                "Actúas como un ingeniero agrónomo experto en {cultivo}. "
                "El usuario tiene una planta con {enfermedad}. "
                "Explícale de forma sencilla qué es la enfermedad, sus síntomas, "
                "y dale 3 soluciones ecológicas o remedios caseros prácticos "
                "para combatirla. Sé empático y directo."
            ).format(cultivo=cultivo, enfermedad=enfermedad),
        }, {
            "role": "user",
            "content": f"Mi planta de {cultivo} tiene: {enfermedad}",
        }],
    )
    return respuesta.choices[0].message.content


def generar_audio_gtts(texto):
    if not _gtts_available:
        return None
    tts = gTTS(text=texto, lang="es")
    buffer = io.BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    return buffer


# ---- Model loading -------------------------------------------------------
@st.cache_resource
def load_crop_model(crop_name):
    model = build_model(get_num_classes(crop_name), weights=None)
    model.load_state_dict(torch.load(model_path(crop_name), map_location="cpu", weights_only=True))
    model.eval()
    return model


# ---- Streamlit UI --------------------------------------------------------
st.set_page_config(page_title="CropGuard", layout="wide")
st.title("CropGuard")
st.markdown("### AI-Powered Crop Disease Diagnosis for Smallholder Farmers")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Leaf Photo")
    crop_list = list(CROP_CLASSES.keys())
    crop = st.selectbox("Select crop", crop_list)
    uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image")

with col2:
    st.subheader("Diagnosis")
    if uploaded is None:
        st.info("Upload a photo to begin diagnosis.")
        st.markdown("---")
        st.markdown("**Supported Crops:**")
        for crop_name, classes in CROP_CLASSES.items():
            st.markdown(f"- **{crop_name}** ({len(classes)} diseases)")
    else:
        num_classes = get_num_classes(crop)
        class_names = CROP_CLASSES[crop]

        if not os.path.exists(model_path(crop)):
            st.error(f"Model for **{crop}** not found ({model_path(crop)}).")
            st.info("Run `python train.py` to train classifiers.")
        else:
            with st.spinner(f"Analyzing {crop} leaf..."):
                model = load_crop_model(crop)

                img_tensor = EVAL_TRANSFORM(img.convert("RGB")).unsqueeze(0)

                with torch.no_grad():
                    logits = model(img_tensor)
                    preds_1d = torch.softmax(logits, dim=1)[0].numpy()

                pred_class = int(preds_1d.argmax())
                conf = float(preds_1d[pred_class])

                disease_name = get_disease_name(crop, pred_class)

            st.success(f"**{disease_name}**  ({conf:.1%} confidence)")
            st.caption(f"Crop: {crop}")

            # Confidence bar chart
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")

            colors = ["#4A90D9" if i == pred_class else "#E0E0E0" for i in range(num_classes)]
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(class_names, preds_1d, color=colors)
            ax.set_xlabel("Confidence")
            ax.set_xlim(0, 1)
            plt.tight_layout()
            st.pyplot(fig)

            # ---- LLM Treatment Advice ----------------------------------------
            st.markdown("---")
            st.markdown("### AI Treatment Advice")

            llm_key = f"tratamiento_llm_{crop}"
            if st.button(f"Get AI advice for {crop} — {disease_name}", type="primary"):
                with st.spinner("Consulting agronomist AI..."):
                    st.session_state[llm_key] = obtener_tratamiento_llm(disease_name, crop)

            if llm_key in st.session_state:
                st.info(st.session_state[llm_key])

                if _gtts_available:
                    if st.button("Listen (TTS)"):
                        with st.spinner("Generating audio..."):
                            audio_buffer = generar_audio_gtts(st.session_state[llm_key])
                        if audio_buffer:
                            st.audio(audio_buffer, format="audio/mp3")

st.markdown("---")
st.caption("CropGuard - Samsung Innovation Campus Capstone Project")
