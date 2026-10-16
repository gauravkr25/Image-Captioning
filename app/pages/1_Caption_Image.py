from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

from app.components import format_caption_result, get_bundle, get_run_names, get_sample_manifest
from src.inference import caption_image


st.title("Caption Image")
with st.sidebar:
    run_name = st.selectbox("Model run", get_run_names())
    beam_size = st.slider("Beam size", 1, 5, 3)

bundle = get_bundle(run_name)

manifest = get_sample_manifest()
if manifest:
    sample_labels = ["Upload image", *[item["scene"] for item in manifest]]
else:
    sample_labels = ["Upload image"]

sample_choice = st.selectbox("Try a sample", sample_labels)

image = None
if sample_choice != "Upload image":
    sample_item = next(item for item in manifest if item["scene"] == sample_choice)
    image_path = f"artifacts/samples/{sample_item['image']}"
    image = Image.open(image_path).convert("RGB")
else:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        if uploaded.size > 10 * 1024 * 1024:
            st.error("Uploaded image exceeds 10 MB. Please choose a smaller file.")
        else:
            try:
                image = Image.open(BytesIO(uploaded.getvalue())).convert("RGB")
            except Exception:
                st.error("The uploaded file could not be decoded as an image.")

if image is not None:
    result = caption_image(bundle, image, beam_size=beam_size, return_attention=False)
    st.image(image, caption="Input image", use_container_width=True)
    st.markdown(f"## {format_caption_result(result)}")
    st.caption(f"Beam candidates: {len(result.get('candidates', []))}")
    candidate_rows = [{"Caption": caption, "Score": float(score)} for caption, score, _ in result.get("candidates", [])[:5]]
    if candidate_rows:
        st.dataframe(pd.DataFrame(candidate_rows), use_container_width=True)
else:
    st.info("Upload an image to generate a caption or select a sample.")
