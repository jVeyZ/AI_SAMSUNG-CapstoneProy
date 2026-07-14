"""
CropGuard - Streamlit Interactive Demo
Tomato Disease Diagnosis for Smallholder Farmers
"""
import os, sys, json, subprocess, tempfile, traceback

import streamlit as st
st.set_page_config(page_title="CropGuard")

from PIL import Image
from treatment_db import TREATMENTS

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(WORK_DIR, "cropguard_model.keras")
PREDICT_WORKER = os.path.join(WORK_DIR, "predict_worker.py")
display_names = [t["name"] for t in TREATMENTS.values()]


st.title("CropGuard")
st.markdown("### AI-Powered Tomato Disease Diagnosis for Smallholder Farmers")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Leaf Photo")
    uploaded = st.file_uploader("Choose a tomato leaf image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image")

with col2:
    st.subheader("Diagnosis")
    if uploaded is None:
        st.info("Upload a photo to begin diagnosis.")
        st.markdown("---")
        st.markdown("**Supported Diseases:**")
        for t in TREATMENTS.values():
            st.markdown(f"- {t['name']}")
    elif not os.path.exists(MODEL_PATH):
        st.error("Model not found. Run train.py first.")
    else:
        if st.button("Classify"):
            with st.spinner("Analyzing..."):
                try:
                    # Save uploaded image to a temp file for the worker
                    suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=WORK_DIR) as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = tmp.name

                    # Run prediction in a separate process (avoids Keras import issues in Streamlit thread)
                    env = os.environ.copy()
                    env["KERAS_BACKEND"] = "torch"
                    proc = subprocess.run(
                        [sys.executable, PREDICT_WORKER, tmp_path],
                        capture_output=True,
                        text=True,
                        cwd=WORK_DIR,
                        env=env,
                        timeout=120,
                    )

                    os.unlink(tmp_path)

                    if proc.returncode != 0:
                        raise RuntimeError(f"Worker failed: {proc.stderr or proc.stdout}")

                    result = json.loads(proc.stdout)
                    if not result.get("ok"):
                        raise RuntimeError(result.get("error", "Unknown error"))

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    with st.expander("Details"):
                        st.code(traceback.format_exc())
                    st.stop()

            pred_class = result["pred_class"]
            conf = result["conf"]
            preds = result["preds"]
            disease = TREATMENTS[pred_class]

            st.success(f"**{disease['name']}** ({conf:.1%} confidence)")

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ['#4A90D9' if i == pred_class else '#E0E0E0'
                      for i in range(len(display_names))]
            ax.barh(display_names, preds, color=colors)
            ax.set_xlabel('Confidence')
            ax.set_xlim(0, 1)
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
