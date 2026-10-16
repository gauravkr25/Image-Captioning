from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.inference import attention_overlay, load_bundle


@st.cache_resource
def get_run_names():
    model_root = Path("artifacts/models")
    if not model_root.exists():
        return ["smoke"]
    runs = sorted([p.name for p in model_root.iterdir() if p.is_dir()])
    return runs or ["smoke"]


@st.cache_resource
def get_bundle(run_name: str):
    return load_bundle(run_name)


@st.cache_data
def get_metrics(run_name: str):
    metrics_path = Path("artifacts/reports") / run_name / "metrics.json"
    if not metrics_path.exists():
        return {}
    import json

    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def get_samples(run_name: str):
    samples_path = Path("artifacts/reports") / run_name / "samples.json"
    if not samples_path.exists():
        return {"samples": []}
    import json

    with samples_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def app_header():
    st.title("Image Captioning Dashboard")
    st.caption("Captions come from a small model trained on a limited image set and can be imperfect on unusual scenes.")


def format_caption_result(result):
    caption = result.get("caption", "")
    return caption if caption else "No caption generated"


def render_attention_for_word(image, attention_map, word_index, word_label, alpha=0.5):
    if attention_map is None:
        return
    overlay = attention_overlay(image, attention_map, alpha=alpha)
    st.image(overlay, caption=f"{word_index + 1}. {word_label}", use_container_width=True)


def preview_top_words(tokenizer, top_n=10):
    if tokenizer is None or not hasattr(tokenizer, "word_to_id"):
        return pd.DataFrame({"word": [], "count": []})
    counts = {word: idx for word, idx in tokenizer.word_to_id.items() if not word.startswith("<")}
    rows = [{"word": word, "count": rank + 1} for word, rank in sorted(counts.items(), key=lambda item: item[1])[:top_n]]
    return pd.DataFrame(rows)


@st.cache_data
def get_sample_manifest():
    manifest_path = Path("artifacts/samples/manifest.json")
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def get_sample_image(label: str):
    image_path = Path("artifacts/samples") / f"{label}.png"
    if not image_path.exists():
        return None
    with Image.open(image_path) as image:
        return image.copy()
