from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.components import get_bundle, get_run_names, get_sample_manifest, get_samples


st.title("Dataset and Vocab")
st.write("Flickr8k overview and vocabulary summary")

run_name = st.selectbox("Model run", get_run_names())
bundle = get_bundle(run_name)
tokenizer = bundle.get("tokenizer")

if tokenizer is not None and hasattr(tokenizer, "word_to_id"):
    vocab_size = len(tokenizer.word_to_id)
    st.metric("Vocabulary size", vocab_size)
    words = [word for word in tokenizer.word_to_id if not word.startswith("<")]
    if words:
        counts = pd.Series({word: idx for idx, word in enumerate(words[:20])})
        st.bar_chart(counts)
else:
    st.info("Tokenizer data is not available for this bundle yet.")

sample_manifest = get_sample_manifest()
if sample_manifest:
    st.subheader("Sample gallery")
    cols = st.columns(3)
    for idx, item in enumerate(sample_manifest[:6]):
        with cols[idx % 3]:
            image_path = Path("artifacts/samples") / item["image"]
            st.image(str(image_path), caption=item["scene"], use_container_width=True)
            st.caption(item["caption"])

samples = get_samples(run_name)
rows = samples.get("samples", [])
if rows:
    sample_df = pd.DataFrame(rows)
    st.subheader("Sample results")
    st.dataframe(sample_df[["image_id", "CIDEr", "prediction"]], use_container_width=True)
else:
    st.caption("No samples.json file is available for this run yet.")

st.write("Dataset provenance: Synthetic smoke data is used for validation in this environment; a real Flickr8k release would add the original image files and captions.")
