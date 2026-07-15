"""
CropGuard - Streamlit Interactive Demo
Tomato Disease Diagnosis for Smallholder Farmers
"""
import os, sys, io
os.environ["KERAS_BACKEND"] = "torch"

import streamlit as st
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from keras.models import load_model
from treatment_db import TREATMENTS
from openai import OpenAI
from gtts import gTTS


def obtener_tratamiento_llm(enfermedad):
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["GROQ_API_KEY"]
    )
    respuesta = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Actúas como un ingeniero agrónomo experto en patologías del tomate en España. "
                    "El usuario tiene una planta con {enfermedad}. "
                    "Explícale de forma muy sencilla qué es y dale 3 soluciones ecológicas o remedios caseros prácticos "
                    "para combatirlo. Sé empático y directo"
                ).format(enfermedad=enfermedad),
            },
            {"role": "user", "content": f"Mi planta de tomate tiene: {enfermedad}"},
        ],
    )
    return respuesta.choices[0].message.content


def generar_audio_gtts(texto):
    tts = gTTS(text=texto, lang="es")
    buffer = io.BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    return buffer

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(WORK_DIR, "cropguard_model.keras")
RESNET_PATH = os.path.join(WORK_DIR, "cropguard_resnet50.pth")

@st.cache_resource
def load_resnet():
    resnet = models.resnet50(weights=None)
    resnet.load_state_dict(torch.load(RESNET_PATH, map_location="cpu", weights_only=True))
    resnet.eval()
    return resnet

resnet_model = load_resnet()
resnet_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

CLASS_NAMES = [
    "Bacterial_spot", "Early_blight", "Healthy", "Late_blight",
    "Leaf_Mold", "Septoria_leaf_spot", "Spider_mites",
    "Target_Spot", "Mosaic_virus", "Yellow_Leaf_Curl_Virus"
]

display_names = [t["name"] for t in TREATMENTS.values()]

st.set_page_config(page_title="CropGuard", layout="wide")
st.title("CropGuard")
st.markdown("### AI-Powered Tomato Disease Diagnosis for Smallholder Farmers")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Leaf Photo")
    uploaded = st.file_uploader("Choose a tomato leaf image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True, caption="Uploaded Image")

with col2:
    st.subheader("Diagnosis")
    if uploaded is None:
        st.info("Upload a photo to begin diagnosis.")
        st.markdown("---")
        st.markdown("**Supported Diseases:**")
        for t in TREATMENTS.values():
            st.markdown(f"- {t['name']}")
    else:
        if not os.path.exists(MODEL_PATH):
            st.error("Model not found. Run train.py first.")
        else:
            with st.spinner("Analyzing..."):
                model = load_model(MODEL_PATH)

                img_resized = img.resize((224, 224))
                img_tensor = resnet_transform(img_resized.convert("RGB")).unsqueeze(0)

                with torch.no_grad():
                    features = resnet_model(img_tensor).numpy()

                preds = model.predict(features, verbose=0)

                if preds.ndim == 1:
                    pred_class = np.argmax(preds)
                    conf = preds[pred_class]
                else:
                    pred_class = np.argmax(preds[0])
                    conf = preds[0][pred_class]

                disease = TREATMENTS[pred_class]

            st.success(f"**{disease['name']}** ({conf:.1%} confidence)")

            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")
            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ['#4A90D9' if i == pred_class else '#E0E0E0' for i in range(len(display_names))]
            ax.barh(display_names, preds[0], color=colors)
            ax.set_xlabel('Confidence'); ax.set_xlim(0, 1)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("---")

            st.markdown("#### Symptoms")
            st.write(disease["symptoms"])
            st.markdown("#### Treatment")
            st.write(disease["treatment"])
            st.markdown("#### Prevention")
            st.write(disease["prevention"])

            st.markdown("---")
            st.markdown("### Tratamiento personalizado con IA")

            if st.button("Generar consejo con IA"):
                with st.spinner("Consultando al agrónomo virtual..."):
                    st.session_state["tratamiento_llm"] = obtener_tratamiento_llm(disease["name"])

            if "tratamiento_llm" in st.session_state:
                st.info(st.session_state["tratamiento_llm"])

                if st.button("🔊 Escuchar Diagnóstico"):
                    with st.spinner("Generando audio del diagnóstico..."):
                        audio_buffer = generar_audio_gtts(st.session_state["tratamiento_llm"])
                    st.audio(audio_buffer, format="audio/mp3", start_time=0)

st.markdown("---")
st.caption("CropGuard - Samsung Innovation Campus Capstone Project")
