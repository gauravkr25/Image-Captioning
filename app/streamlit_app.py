import streamlit as st

from app.components import app_header, get_run_names


st.set_page_config(page_title="Image Captioning", page_icon="🖼️")
app_header()

st.write("Use the pages in the sidebar to inspect captions, attention maps, and evaluation summaries.")

with st.sidebar:
    st.selectbox("Model run", get_run_names())
