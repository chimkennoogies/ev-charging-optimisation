import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Allow this module to resolve the packaged project code under `outputs/src`.
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "outputs"))

from src.utils import get_project_root


def _load_metrics() -> Optional[dict]:
    metrics_path = get_project_root() / "outputs" / "test_metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_model_comparison() -> Optional[dict]:
    comparison_path = get_project_root() / "outputs" / "model_comparison_metrics.json"
    if not comparison_path.exists():
        return None
    with open(comparison_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_processed_data() -> Optional[pd.DataFrame]:
    processed_path = get_project_root() / "data" / "processed" / "processed_data.csv"
    if not processed_path.exists():
        return None
    return pd.read_csv(processed_path)


def render_results_tab(model_name: str = "Best Trained Model") -> None:
    """Render experiment outputs in a dedicated Streamlit tab."""
    st.subheader("Experiment Results")
    st.caption("Metrics, dataset profile, diagnostics, and experiment configuration.")

    metrics = _load_metrics()
    df_processed = _load_processed_data()

    st.markdown("### Dataset Summary")
    dcol1, dcol2, dcol3 = st.columns(3)

    total_sessions = int(len(df_processed)) if df_processed is not None else 0
    feature_count = int(df_processed.shape[1] - 1) if df_processed is not None and df_processed.shape[1] > 0 else 0
    date_range = "N/A"
    missing_values = "N/A"

    if df_processed is not None:
        missing_values = int(df_processed.isna().sum().sum())
        if "Charging Start Time" in df_processed.columns:
            times = pd.to_datetime(df_processed["Charging Start Time"], errors="coerce")
            if times.notna().any():
                date_range = f"{times.min().date()} to {times.max().date()}"

    dcol1.metric("Total Sessions", f"{total_sessions:,}")
    dcol2.metric("Feature Columns", f"{feature_count:,}")
    dcol3.metric("Total Missing Values", str(missing_values))
    st.caption(f"Date range: {date_range}")

    st.markdown("### Model Performance")
    if metrics is None:
        st.warning("No metrics found. Run `python main.py` to generate experiment outputs.")
    else:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Battery Degradation", f"{metrics.get('battery_degradation', 0):.3f}")
        metric_cols[1].metric("Capacity Retention (%)", f"{metrics.get('capacity_retention_pct', 0):.1f}")
        metric_cols[2].metric("Charging Efficiency", f"{metrics.get('charging_efficiency', 0):.3f}")
        metric_cols[3].metric("Thermal Stress Index", f"{metrics.get('thermal_stress_index', 0):.2f}")

        comparison_df = pd.DataFrame(
            [
                {
                    "Model": model_name,
                    "Battery Degradation": metrics.get("battery_degradation"),
                    "Capacity Retention (%)": metrics.get("capacity_retention_pct"),
                    "Charging Efficiency": metrics.get("charging_efficiency"),
                    "Thermal Stress Index": metrics.get("thermal_stress_index"),
                }
            ]
        )
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    model_comparison = _load_model_comparison()
    if model_comparison and "models" in model_comparison:
        st.markdown("### Model Comparison")
        comparison_table = pd.DataFrame(model_comparison["models"])
        st.dataframe(comparison_table, use_container_width=True, hide_index=True)

    st.markdown("### Diagnostic Plots")
    figures_dir = get_project_root() / "outputs" / "figures"
    degradation_plot = figures_dir / "battery_degradation_comparison.png"
    secondary_plot = figures_dir / "secondary_metrics_comparison.png"

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        if degradation_plot.exists():
            st.image(str(degradation_plot), caption="Battery Degradation Comparison")
        else:
            st.info("Missing plot: battery_degradation_comparison.png")

    with pcol2:
        if secondary_plot.exists():
            st.image(str(secondary_plot), caption="Secondary Metrics Comparison")
        else:
            st.info("Missing plot: secondary_metrics_comparison.png")

    st.markdown("### Experiment Configuration")
    with st.expander("Show setup details"):
        st.write("- Split strategy: Chronological train-test split (80/20)")
        st.write("- Models: Random Forest, XGBoost (best saved as `trained_model.pkl`)")
        st.write("- Evaluation: battery degradation, capacity retention, charging efficiency, thermal stress")
        st.write("- Outputs: `outputs/test_metrics.json`, `outputs/model_comparison_metrics.json`")
