"""
CropGuard - Streamlit Interactive Demo
Tomato Disease Diagnosis for Smallholder Farmers
"""
import os, sys
os.environ["KERAS_BACKEND"] = "torch"

import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from treatment_db import TREATMENTS

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(WORK_DIR, "cropguard_model.keras")
RESNET_PATH = os.path.join(WORK_DIR, "cropguard_resnet50.pth")

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
                img_array = np.array(img_resized).astype(np.float32) / 255.0

                if img_array.ndim == 2:
                    img_array = np.stack([img_array]*3, axis=-1)

                img_batch = np.expand_dims(img_array, 0)
                preds = model.predict(img_batch, verbose=0)

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
st.caption("CropGuard - Samsung Innovation Campus Capstone Project")
