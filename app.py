"""
CropGuard - Streamlit Interactive Demo
Multi-Crop Disease Diagnosis (Tomato, Rice, Orange)
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

CROP_DATA = {}
for _id, _info in TREATMENTS.items():
    _crop = _info["crop"]
    if _crop not in CROP_DATA:
        CROP_DATA[_crop] = {"classes": [], "treatment_start": _id, "data_dir": _crop.lower()}
    CROP_DATA[_crop]["classes"].append(_info["name"])


def obtener_tratamiento_llm(enfermedad, cultivo):
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
                    "Actúas como un ingeniero agrónomo experto en patologías de {cultivo} en España. "
                    "El usuario tiene una planta con {enfermedad}. "
                    "Explícale de forma muy sencilla qué es y dale 3 soluciones ecológicas o remedios caseros prácticos "
                    "para combatirlo. Sé empático y directo"
                ).format(cultivo=cultivo, enfermedad=enfermedad),
            },
            {"role": "user", "content": f"Mi planta de {cultivo} tiene: {enfermedad}"},
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

@st.cache_resource
def load_resnet():
    resnet_path = os.path.join(WORK_DIR, "cropguard_resnet50.pth")
    resnet = models.resnet50(weights=None)
    resnet.load_state_dict(torch.load(resnet_path, map_location="cpu", weights_only=True))
    resnet.eval()
    return resnet

resnet_model = load_resnet()
resnet_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

st.set_page_config(page_title="CropGuard", layout="wide")
st.title("CropGuard")
st.markdown("### AI-Powered Crop Disease Diagnosis for Smallholder Farmers")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Leaf Photo")
    crop = st.selectbox("Tipo cultivo", ["Tipo cultivo"] + list(CROP_DATA.keys()))
    uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True, caption="Uploaded Image")

with col2:
    st.subheader("Diagnosis")
    if uploaded is None or crop == "Tipo cultivo":
        if crop == "Tipo cultivo" and uploaded is not None:
            st.warning("Please select a crop type first.")
        else:
            st.info("Select a crop type and upload a photo to begin diagnosis.")
        st.markdown("---")
        st.markdown("**Supported Crops:**")
        for crop_name, info in CROP_DATA.items():
            st.markdown(f"- **{crop_name}** ({len(info['classes'])} diseases)")
    else:
        crop_info = CROP_DATA[crop]
        num_classes = len(crop_info["classes"])
        treatment_offset = crop_info["treatment_start"]
        class_names = crop_info["classes"]

        model_path = os.path.join(WORK_DIR, f"cropguard_{crop.lower()}_classifier.keras")

        if not os.path.exists(model_path):
            st.error(f"Model for {crop} not found. Run train.py first.")
        else:
            with st.spinner(f"Analyzing {crop} leaf..."):
                model = load_model(model_path)

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

                treatment_id = treatment_offset + pred_class
                disease = TREATMENTS[treatment_id]

            st.success(f"**{disease['name']}** ({conf:.1%} confidence)")
            st.caption(f"Crop: {crop}")

            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")

            crop_display_names = [TREATMENTS[treatment_offset + i]["name"] for i in range(num_classes)]
            probs = preds[0] if preds.ndim > 1 else preds
            colors = ['#4A90D9' if i == pred_class else '#E0E0E0' for i in range(num_classes)]

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(crop_display_names, probs, color=colors)
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

            llm_key = f"tratamiento_llm_{crop}"
            if st.button("Generar consejo con IA"):
                with st.spinner("Consultando al agrónomo virtual..."):
                    st.session_state[llm_key] = obtener_tratamiento_llm(disease["name"], crop)

            if llm_key in st.session_state:
                st.info(st.session_state[llm_key])

                if st.button("🔊 Escuchar Diagnóstico"):
                    with st.spinner("Generando audio del diagnóstico..."):
                        audio_buffer = generar_audio_gtts(st.session_state[llm_key])
                    st.audio(audio_buffer, format="audio/mp3", start_time=0)

st.markdown("---")
st.caption("CropGuard - Samsung Innovation Campus Capstone Project")
