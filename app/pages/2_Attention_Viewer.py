from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image

from app.components import get_bundle, get_run_names
from src.inference import attention_overlay, caption_image


st.title("Attention Viewer")
st.write("Where was the model looking when it wrote this word?")

with st.sidebar:
    run_name = st.selectbox("Model run", get_run_names())
    beam_size = st.slider("Beam size", 1, 5, 3)
    colormap = st.selectbox("Heatmap colormap", ["viridis", "plasma", "magma", "inferno", "cividis"])
    alpha = st.slider("Heatmap opacity", 0.2, 0.9, 0.55, step=0.05)

bundle = get_bundle(run_name)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
image = None
if uploaded is not None:
    if uploaded.size > 10 * 1024 * 1024:
        st.error("Uploaded image exceeds 10 MB. Please choose a smaller file.")
    else:
        try:
            image = Image.open(BytesIO(uploaded.getvalue())).convert("RGB")
        except Exception:
            st.error("The uploaded file could not be decoded as an image.")

if image is None:
    st.info("Upload an image to inspect its attention maps.")
    st.stop()

result = caption_image(bundle, image, beam_size=beam_size, return_attention=True)
candidate_caption = result.get("caption", "")
words = [word for word in candidate_caption.split() if word]
if not words:
    st.warning("The generated caption is empty, so there are no attention maps to show.")
    st.stop()

word_index = st.slider("Select word", 0, max(len(words) - 1, 0), 0)
attention_map = result.get("token_attention", [None] * len(words))[word_index]
if attention_map is None:
    st.warning("Attention for this token is unavailable.")
    st.stop()

st.markdown(f"### Word {word_index + 1}: **{words[word_index]}**")
st.caption(f"Heatmap style: {colormap}; opacity: {alpha:.2f}")
panel = attention_overlay(image, attention_map, alpha=alpha, colormap=colormap)
st.image(panel, caption=f"Attention for '{words[word_index]}'", use_container_width=True)

st.caption("The model inferred this caption: " + candidate_caption)
