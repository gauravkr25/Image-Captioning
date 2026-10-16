from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.components import get_metrics, get_run_names


st.title("Model Comparison")
st.write("GRU baseline vs Transformer")

runs = get_run_names()
selected_runs = st.multiselect("Runs to compare", runs, default=runs[: min(2, len(runs))])
if not selected_runs:
    st.info("Select at least one model run to compare.")
    st.stop()

metric_keys = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "CIDEr"]
rows = []
for run_name in selected_runs:
    metrics = get_metrics(run_name)
    if metrics:
        row = {"run": run_name}
        for key in metric_keys:
            row[key] = float(metrics.get(key, 0.0))
        rows.append(row)

if rows:
    metrics_df = pd.DataFrame(rows).set_index("run")
    st.dataframe(metrics_df, use_container_width=True)
else:
    st.warning("No evaluation metrics are available for the selected run(s).")

for run_name in selected_runs:
    history_path = Path("artifacts/reports") / run_name / "history.csv"
    if history_path.exists():
        st.subheader(f"Training curve for {run_name}")
        try:
            history_df = pd.read_csv(history_path)
        except Exception:
            history_df = pd.DataFrame()
        if not history_df.empty:
            st.line_chart(history_df.set_index("epoch"))
        else:
            st.caption("No training history is available for this run yet.")
