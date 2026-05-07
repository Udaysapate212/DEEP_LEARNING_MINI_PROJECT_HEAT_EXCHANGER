from __future__ import annotations

from pathlib import Path
from textwrap import dedent
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
VERSION1_DIR = ROOT_DIR / "version_1"
if str(VERSION1_DIR) not in sys.path:
    sys.path.insert(0, str(VERSION1_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from heat_exchanger_best_model import FLUID_PRESETS, predict_scenario

try:
    from physics_informed_lstm import PhysicsInformedLSTM

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from version_2.study_utils import (
    HOT_OUTLET_TARGET,
    PRIMARY_TARGET,
    RESULTS_DIR,
    SECONDARY_TARGET,
    TARGET_LABELS,
    TARGET_UNITS,
    load_sampling_manifest,
)

# Version 2 dashboard loads low-data artifacts from disk and performs only inference.
# Training is separated into `python version_2/run_low_data_study.py`.

METRICS_PATH = RESULTS_DIR / "low_data_metrics.csv"
PREDICTIONS_PATH = RESULTS_DIR / "low_data_predictions.csv"
BEST_MODELS_PATH = RESULTS_DIR / "low_data_best_models.csv"
FULL_ARTIFACT_PATH = VERSION1_DIR / "best_heat_exchanger_models.joblib"
FULL_PILSTM_ARTIFACT_PATH = VERSION1_DIR / "pilstm_artifact.joblib"
FULL_PILSTM_WEIGHTS_PATH = VERSION1_DIR / "pilstm_model.weights.h5"


st.set_page_config(
    page_title="Version 2 Low-Data Digital Twin",
    page_icon="V2",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_joblib(path: Path) -> dict | None:
    if not path.exists():
        return None
    return joblib.load(path)


def inject_styles() -> None:
    st.markdown(
        dedent(
            """
            <style>
            .stApp { background: linear-gradient(135deg, #f4efe6 0%, #f6f1e8 35%, #eaf2ef 100%); }
            .stMarkdown, .main .block-container, .main .block-container * { color: #183642 !important; }
            .hero-card, .prediction-card { background: rgba(255,255,255,0.9); border: 1px solid rgba(24,54,66,0.10); border-radius: 18px; box-shadow: 0 12px 30px rgba(38,56,64,0.08); color: #183642; }
            .hero-card { padding: 1.25rem 1.4rem; margin-bottom: 1rem; }
            .hero-title { font-size: 2rem; font-weight: 800; color: #183642; }
            .hero-copy { color: #284650; line-height: 1.6; margin-top: 0.4rem; }
            .chip { display: inline-block; margin: 0.3rem 0.35rem 0 0; padding: 0.28rem 0.68rem; border-radius: 999px; font-size: 0.82rem; font-weight: 700; color: #fff; background: #173f4f; }
            .chip.warm { background: #a54819; }
            .chip.teal { background: #0f766e; }
            .chip.blue { background: #1d4ed8; }
            .chip.purple { background: #7c3aed; }
            .prediction-card { padding: 1rem; height: 100%; }
            .prediction-label { font-size: 0.82rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #35515b; margin-bottom: 0.7rem; }
            .prediction-value { font-size: 2.2rem; font-weight: 800; line-height: 1.05; color: #183642; }
            .prediction-subvalue { font-size: 0.98rem; font-weight: 700; color: #32515b; }
            .prediction-note { margin-top: 0.55rem; font-size: 0.9rem; color: #49626b; line-height: 1.45; }
            </style>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def kelvin_to_celsius(value_k: float) -> float:
    return value_k - 273.15


def format_temperature(value_k: float | None) -> str:
    if value_k is None:
        return "Not available"
    return f"{value_k:.3f} K / {kelvin_to_celsius(value_k):.3f} deg C"


def format_delta(value: float | None, unit: str) -> str:
    if value is None:
        return "Not available"
    return f"{value:+.3f} {unit}"


def render_prediction_card(column, label: str, value: str, note: str, subvalue: str = "") -> None:
    column.markdown(
        dedent(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">{label}</div>
                <div class="prediction-value">{value}</div>
                <div class="prediction-subvalue">{subvalue}</div>
                <div class="prediction-note">{note}</div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def restore_pilstm(artifact_path: Path, weights_path: Path) -> dict | None:
    if not TENSORFLOW_AVAILABLE or not artifact_path.exists() or not weights_path.exists():
        return None
    try:
        artifact = joblib.load(artifact_path)
        pilstm = PhysicsInformedLSTM(
            sequence_length=artifact["sequence_length"],
            lstm_units=artifact["lstm_units"],
            learning_rate=artifact["learning_rate"],
        )
        pilstm.build_model(input_shape=(artifact["sequence_length"], 8))
        pilstm.model.load_weights(str(weights_path))
        pilstm.scaler_X.mean_ = artifact["scaler_X_mean"]
        pilstm.scaler_X.scale_ = artifact["scaler_X_scale"]
        pilstm.scaler_y.mean_ = artifact["scaler_y_mean"]
        pilstm.scaler_y.scale_ = artifact["scaler_y_scale"]
        artifact["pi_lstm"] = pilstm
        return artifact
    except Exception:
        return None


def derive_heat_load(hot_inlet_temp: float, hot_outlet_temp: float, hot_props: dict[str, float]) -> float:
    hot_cp = hot_props.get("cp_kj_kgk", 4.18)
    hot_mass_flow = 1.0
    return hot_mass_flow * hot_cp * (hot_inlet_temp - hot_outlet_temp)


def predict_pilstm(pilstm_artifact: dict, hot_inlet: float, cold_flow: float, heat_load: float) -> np.ndarray | None:
    try:
        pilstm = pilstm_artifact["pi_lstm"]
        lmtd_estimate = (hot_inlet - 293.15) * 0.6
        row = [hot_inlet, cold_flow, heat_load, 500000.0, 100000.0, 1.0, cold_flow, lmtd_estimate]
        seq = pd.DataFrame([row] * pilstm.sequence_length).to_numpy(dtype=float).reshape(1, pilstm.sequence_length, -1)
        pred = pilstm.predict(seq)[0]
        if pred.shape[0] != 2:
            return None
        if pred[0] > hot_inlet or pred[0] < 293.15:
            return None
        return np.asarray(pred, dtype=float)
    except Exception:
        return None


def fluid_block(title: str, prefix: str) -> dict[str, float]:
    preset_name = st.selectbox(f"{title} preset", [*FLUID_PRESETS.keys(), "Custom"], key=f"{prefix}_preset")
    preset = FLUID_PRESETS.get(preset_name, FLUID_PRESETS["Water"])
    c1, c2 = st.columns(2)
    with c1:
        cp = st.number_input(f"{title} Cp", min_value=0.1, max_value=10.0, value=float(preset["cp_kj_kgk"]), step=0.01, key=f"{prefix}_cp")
        rho = st.number_input(f"{title} density", min_value=1.0, max_value=2500.0, value=float(preset["rho_kg_m3"]), step=1.0, key=f"{prefix}_rho")
    with c2:
        mu = st.number_input(f"{title} viscosity", min_value=0.00001, max_value=5.0, value=float(preset["mu_pa_s"]), step=0.0001, format="%.5f", key=f"{prefix}_mu")
        k = st.number_input(f"{title} conductivity", min_value=0.01, max_value=5.0, value=float(preset["k_w_mk"]), step=0.01, key=f"{prefix}_k")
    return {"name": preset_name, "cp_kj_kgk": cp, "rho_kg_m3": rho, "mu_pa_s": mu, "k_w_mk": k}


def artifact_metric_lookup(artifact: dict | None) -> dict[str, dict[str, float]]:
    if artifact is None:
        return {}
    return {row["Target"]: row for row in artifact.get("metrics", []) if "Target" in row}


def best_row(best_df: pd.DataFrame, subset_name: str, target: str) -> pd.Series:
    return best_df.loc[(best_df["subset_name"] == subset_name) & (best_df["target"] == target)].iloc[0]


def full_artifact_model_name(artifact: dict | None, full_row: dict[str, float]) -> str:
    if "Model" in full_row:
        return str(full_row["Model"])
    if artifact is None:
        return "Not available"
    best_family = artifact.get("best_model_family")
    if isinstance(best_family, str):
        return best_family
    if isinstance(best_family, dict):
        return str(best_family)
    return "Not available"


inject_styles()
metrics_df = load_csv(METRICS_PATH)
predictions_df = load_csv(PREDICTIONS_PATH)
best_models_df = load_csv(BEST_MODELS_PATH)
manifest_df = load_sampling_manifest()

if metrics_df is None or predictions_df is None or best_models_df is None:
    st.warning("Study results are missing. Run `python version_2/run_low_data_study.py` first.")
    st.stop()

subset_sizes = [100]
selected_subset = 100
selected_subset_name = "low_data_100"

with st.sidebar:
    st.header("Version 2 Inputs")
    st.markdown("**Demonstration subset:** 100 rows from the low-data study.")
    manifest_row = None
    if manifest_df is not None:
        rows = manifest_df.loc[manifest_df["subset_size"] == selected_subset]
        if not rows.empty:
            manifest_row = rows.iloc[0]
            st.caption(
                f"Coverage: {manifest_row['temp_min_k']:.2f} K to {manifest_row['temp_max_k']:.2f} K with "
                f"{int(manifest_row['unique_temp_values'])} unique temperature levels."
            )
    st.caption("Same Version 1 pipeline, but trained only on the selected low-data subset.")
    hot_fluid = fluid_block("Hot fluid", "hot")
    cold_fluid = fluid_block("Cold fluid", "cold")
    st.markdown("### Temperature Inputs")
    hot_inlet_temperature_k = st.slider("Hot inlet temperature (K)", 320.0, 650.0, 473.15, 1.0)
    cold_inlet_temperature_k = st.slider("Cold inlet temperature (K)", 280.0, 350.0, 293.15, 1.0, help="Temperature of cold fluid entering")
    st.markdown("### Flow Rate Inputs")
    cold_inlet_mass_flow_kg_s = st.slider("Cold inlet mass flow (kg/s)", 0.30, 6.00, 2.75, 0.01)
    st.markdown("### Sensor Noise (Optional)")
    hot_sensor_bias_k = st.slider("Hot temperature sensor bias (K)", -20.0, 20.0, 0.0, 0.1)
    cold_flow_sensor_bias_kg_s = st.slider("Cold flow sensor bias (kg/s)", -0.50, 0.50, 0.0, 0.01)
    run_prediction = st.button("Run Low-Data Prediction", use_container_width=True, type="primary")

artifact = load_joblib(RESULTS_DIR / f"{selected_subset_name}_best_models.joblib")
low_pilstm = restore_pilstm(
    RESULTS_DIR / f"{selected_subset_name}_pilstm_artifact.joblib",
    RESULTS_DIR / f"{selected_subset_name}_pilstm.weights.h5",
)
full_artifact = load_joblib(FULL_ARTIFACT_PATH)
full_pilstm = restore_pilstm(FULL_PILSTM_ARTIFACT_PATH, FULL_PILSTM_WEIGHTS_PATH)

if artifact is None:
    st.warning(f"Artifact for `{selected_subset_name}` is missing. Run `python version_2/run_low_data_study.py` first.")
    st.stop()

heat_best = best_row(best_models_df, selected_subset_name, PRIMARY_TARGET)
hot_best = best_row(best_models_df, selected_subset_name, SECONDARY_TARGET)
chips = [
    f'<span class="chip">{selected_subset} rows</span>',
    f'<span class="chip warm">{heat_best["best_model"]} for heat load</span>',
    f'<span class="chip teal">{hot_best["best_model"]} for hot outlet</span>',
    '<span class="chip blue">Deep baseline: PI-LSTM</span>',
]
if manifest_row is not None:
    chips.append(f'<span class="chip blue">{manifest_row["temp_min_k"]:.2f} K to {manifest_row["temp_max_k"]:.2f} K</span>')
if low_pilstm is not None:
    low_accuracy = 100.0 - float(low_pilstm["metrics"]["hot_outlet"]["MAPE"])
    chips.append(f'<span class="chip purple">PI-LSTM hot-outlet accuracy {low_accuracy:.2f}%</span>')

st.markdown(
    dedent(
        f"""
        <div class="hero-card">
            <div class="hero-title">Version 2: Low-Data Heat Exchanger Digital Twin</div>
            <div class="hero-copy">
                This version is intentionally built like Version 1. It keeps the same traditional ML plus hybrid
                prediction flow and adds PI-LSTM as the advanced hot-outlet comparison model. The key difference is
                that this demonstration is fixed to the 100-row low-data subset, so the dashboard shows the real
                low-data workflow clearly and consistently.
            </div>
            <div style="margin-top:0.7rem;">{''.join(chips)}</div>
        </div>
        """
    ).strip(),
    unsafe_allow_html=True,
)

if not run_prediction:
    st.info("Choose the low-data subset and scenario inputs in the sidebar, then click `Run Low-Data Prediction`.")
    st.stop()

results = predict_scenario(
    artifact=artifact,
    hot_inlet_temperature_k=hot_inlet_temperature_k,
    hot_inlet_temperature_k_noisy=hot_inlet_temperature_k + hot_sensor_bias_k,
    cold_inlet_temperature_k=cold_inlet_temperature_k,
    cold_inlet_mass_flow_kg_s=cold_inlet_mass_flow_kg_s,
    cold_inlet_mass_flow_kg_s_noisy=cold_inlet_mass_flow_kg_s + cold_flow_sensor_bias_kg_s,
    hot_props=hot_fluid,
    cold_props=cold_fluid,
)
pilstm_pred = None if low_pilstm is None else predict_pilstm(
    low_pilstm,
    hot_inlet=hot_inlet_temperature_k,
    cold_flow=cold_inlet_mass_flow_kg_s,
    heat_load=results["predicted_heat_load_kw_hybrid"],
)
full_results = None
full_pilstm_hot_outlet = None
if full_artifact is not None:
    full_results = predict_scenario(
        artifact=full_artifact,
        hot_inlet_temperature_k=hot_inlet_temperature_k,
        hot_inlet_temperature_k_noisy=hot_inlet_temperature_k + hot_sensor_bias_k,
        cold_inlet_temperature_k=cold_inlet_temperature_k,
        cold_inlet_mass_flow_kg_s=cold_inlet_mass_flow_kg_s,
        cold_inlet_mass_flow_kg_s_noisy=cold_inlet_mass_flow_kg_s + cold_flow_sensor_bias_kg_s,
        hot_props=hot_fluid,
        cold_props=cold_fluid,
    )
if full_results is not None and full_pilstm is not None:
    full_pilstm_pred = predict_pilstm(
        full_pilstm,
        hot_inlet_temperature_k,
        cold_inlet_mass_flow_kg_s,
        full_results["predicted_heat_load_kw_hybrid"],
    )
    if full_pilstm_pred is not None and len(np.asarray(full_pilstm_pred).reshape(-1)) >= 1:
        full_pilstm_hot_outlet = float(np.asarray(full_pilstm_pred).reshape(-1)[0])

st.subheader("Prediction Snapshot")
c1, c2, c3 = st.columns(3)
render_prediction_card(c1, "Low-data ML heat load", f"{results['predicted_heat_load_kw_ml']:.2f} kW", "Best Version 1-style traditional model on the selected subset.")
render_prediction_card(c2, "Low-data hybrid heat load", f"{results['predicted_heat_load_kw_hybrid']:.2f} kW", "Fluid-aware correction layer applied after the low-data ML prediction.")
render_prediction_card(c3, "Low-data ML hot outlet", f"{results['predicted_hot_outlet_k_ml']:.2f} K", "Traditional low-data hot-outlet estimate.", f"{kelvin_to_celsius(results['predicted_hot_outlet_k_ml']):.2f} deg C")

c4, c5, c6 = st.columns(3)
render_prediction_card(c4, "Low-data hybrid hot outlet", f"{results['predicted_hot_outlet_k_hybrid']:.2f} K", "Hot-stream exit after fluid adjustment.", f"{kelvin_to_celsius(results['predicted_hot_outlet_k_hybrid']):.2f} deg C")
render_prediction_card(c5, "Low-data ML cold outlet", f"{results['predicted_cold_outlet_k_ml']:.2f} K", "Cold-stream exit temperature predicted by ML model.", f"{kelvin_to_celsius(results['predicted_cold_outlet_k_ml']):.2f} deg C")
render_prediction_card(c6, "Low-data hybrid cold outlet", f"{results['predicted_cold_outlet_k_hybrid']:.2f} K", "Cold-stream exit after fluid adjustment.", f"{kelvin_to_celsius(results['predicted_cold_outlet_k_hybrid']):.2f} deg C")

def format_model_outputs(pred: np.ndarray | float | int | None, model_label: str) -> tuple[str, str, str]:
    if pred is None:
        return "N/A", "N/A", "N/A"
    if isinstance(pred, (float, int)):
        hot_outlet = float(pred)
        heat_load = derive_heat_load(hot_inlet_temperature_k, hot_outlet, hot_fluid)
        return (
            f"{hot_outlet:.2f} K",
            "N/A",
            f"{heat_load:.2f} kW",
        )
    pred_arr = np.asarray(pred, dtype=float).reshape(-1)
    if pred_arr.size == 1:
        hot_outlet = float(pred_arr[0])
        heat_load = derive_heat_load(hot_inlet_temperature_k, hot_outlet, hot_fluid)
        return (
            f"{hot_outlet:.2f} K",
            "N/A",
            f"{heat_load:.2f} kW",
        )
    if pred_arr.size != 2:
        return "N/A", "N/A", "N/A"
    hot_outlet, cold_outlet = float(pred_arr[0]), float(pred_arr[1])
    heat_load = derive_heat_load(hot_inlet_temperature_k, hot_outlet, hot_fluid)
    return (
        f"{hot_outlet:.2f} K",
        f"{cold_outlet:.2f} K",
        f"{heat_load:.2f} kW",
    )

pilstm_hot, pilstm_cold, pilstm_heat = format_model_outputs(pilstm_pred, "PI-LSTM")

st.subheader("Deep model predictions")
row1, row2, row3 = st.columns(3)
render_prediction_card(row1, "Low-data PI-LSTM hot outlet", pilstm_hot, "Sequence-aware comparison model with the current project PI-LSTM setup.", "")
render_prediction_card(row2, "Low-data PI-LSTM cold outlet", pilstm_cold, "Sequence-aware comparison model with the current project PI-LSTM setup.", "")
render_prediction_card(row3, "Low-data PI-LSTM heat load", pilstm_heat, "Derived heat load from PI-LSTM hot-outlet prediction.", "")

if full_results is not None:
    st.subheader("Current Shift Vs Version 1")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric(
        "ML heat-load shift",
        format_delta(results["predicted_heat_load_kw_ml"] - full_results["predicted_heat_load_kw_ml"], "kW"),
    )
    d2.metric(
        "ML hot-outlet shift",
        format_delta(results["predicted_hot_outlet_k_ml"] - full_results["predicted_hot_outlet_k_ml"], "K"),
    )
    d3.metric(
        "Hybrid hot-outlet shift",
        format_delta(results["predicted_hot_outlet_k_hybrid"] - full_results["predicted_hot_outlet_k_hybrid"], "K"),
    )
    d4.metric(
        "PI-LSTM shift",
        format_delta(None if pilstm_pred is None or full_pilstm_hot_outlet is None else float(pilstm_pred[0]) - full_pilstm_hot_outlet, "K"),
    )
    st.caption(
        "If these shifts are small, that does not mean Version 2 is wrong. It means the reduced subset still spans the same clean operating manifold. "
        "The stronger low-data evidence is in the saved error metrics and the Full vs Low tab."
    )

tabs = st.tabs(["Comparison", "Evidence", "Deep Comparison", "Full vs Low", "Why PI-LSTM"])

with tabs[0]:
    comparison_rows = [
        {
            "Approach": "Traditional ML",
            "Target": "Heat load",
            "Selected model": heat_best["best_model"],
            "Saved RMSE": f"{heat_best['best_rmse']:.4f} kW",
            "Current prediction": f"{results['predicted_heat_load_kw_ml']:.3f} kW",
        },
        {
            "Approach": "Traditional ML",
            "Target": "Hot outlet",
            "Selected model": hot_best["best_model"],
            "Saved RMSE": f"{hot_best['best_rmse']:.4f} K",
            "Current prediction": format_temperature(results["predicted_hot_outlet_k_ml"]),
        },
        {
            "Approach": "Hybrid correction",
            "Target": "Heat load + hot outlet",
            "Selected model": "Fluid-aware layer",
            "Saved RMSE": "Scenario layer",
            "Current prediction": f"{results['predicted_heat_load_kw_hybrid']:.3f} kW and {format_temperature(results['predicted_hot_outlet_k_hybrid'])}",
        },
    ]
    if low_pilstm is not None:
        comparison_rows.append(
            {
                "Approach": "PI-LSTM",
                "Target": "Hot outlet",
                "Selected model": "Physics-Informed LSTM",
                "Saved RMSE": f"{low_pilstm['metrics']['hot_outlet']['RMSE']:.4f} K",
                "Current prediction": pilstm_hot,
            }
        )
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
    st.caption("Version 2 now presents PI-LSTM in the same overall dashboard story as Version 1 instead of as a separate experiment page.")

with tabs[1]:
    selected_target = st.selectbox("Evaluation target", [PRIMARY_TARGET, SECONDARY_TARGET], format_func=lambda x: TARGET_LABELS[x])
    metric_name = st.selectbox("Metric", ["RMSE", "MAE", "MAPE", "R2", "accuracy_proxy"], index=0)
    target_metrics = metrics_df.loc[(metrics_df["subset_name"] == selected_subset_name) & (metrics_df["target"] == selected_target)].copy()
    target_metrics = target_metrics.sort_values(metric_name, ascending=metric_name not in {"R2", "accuracy_proxy"})
    bar = px.bar(target_metrics, x="model", y=metric_name, color="family", text=metric_name, title=f"{metric_name} for {TARGET_LABELS[selected_target]} on the {selected_subset}-row subset")
    bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(bar, use_container_width=True)
    st.dataframe(target_metrics[["model", "family", "split_strategy", "RMSE", "MAE", "MAPE", "R2", "rank_by_rmse", "notes"]], use_container_width=True, hide_index=True)
    pool = predictions_df.loc[(predictions_df["subset_name"] == selected_subset_name) & (predictions_df["target"] == selected_target)].copy()
    selected_model = st.selectbox("Prediction trace", sorted(pool["model"].unique().tolist()))
    view = pool.loc[pool["model"] == selected_model].copy()
    scatter = px.scatter(view, x="actual_value", y="predicted_value", color="hot_inlet_temperature_k", labels={"actual_value": f"Actual ({TARGET_UNITS[selected_target]})", "predicted_value": f"Predicted ({TARGET_UNITS[selected_target]})"}, title=f"Actual vs predicted {TARGET_LABELS[selected_target].lower()}: {selected_model}")
    low_axis = min(view["actual_value"].min(), view["predicted_value"].min())
    high_axis = max(view["actual_value"].max(), view["predicted_value"].max())
    scatter.add_shape(type="line", x0=low_axis, y0=low_axis, x1=high_axis, y1=high_axis)
    st.plotly_chart(scatter, use_container_width=True)
    error = px.line(view, x="point_index", y="error", markers=True, title=f"Prediction error trace: {selected_model}", labels={"error": f"Error ({TARGET_UNITS[selected_target]})"})
    st.plotly_chart(error, use_container_width=True)

with tabs[2]:
    deep_models = ["PI-LSTM"]
    deep_view = metrics_df.loc[
        (metrics_df["subset_name"] == selected_subset_name)
        & (metrics_df["target"] == HOT_OUTLET_TARGET)
        & (metrics_df["model"] == "PI-LSTM")
    ].copy()
    deep_view = deep_view.sort_values(["RMSE", "MAE"], ascending=[True, True])
    st.markdown("### PI-LSTM Low-Data Performance")
    st.write(
        "This view reports the low-data PI-LSTM hot-outlet performance compared to the traditional Version 1 baselines and the low-data study evidence."
    )
    deep_bar = px.bar(
        deep_view,
        x="model",
        y="RMSE",
        color="family",
        text="RMSE",
        title=f"PI-LSTM hot-outlet RMSE on the {selected_subset}-row subset",
    )
    deep_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(deep_bar, use_container_width=True)
    st.dataframe(
        deep_view[["model", "family", "RMSE", "MAE", "MAPE", "rank_by_rmse", "split_strategy", "notes"]],
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        dedent(
            """
            - `PI-LSTM` is the current project sequence-aware physics-informed comparison model.
            - This panel focuses only on the PI-LSTM low-data evidence, not on separate deep-learning baselines.
            - If the reduced dataset still favors traditional tabular structure, the PI-LSTM report will show that directly.
            """
        ).strip()
    )

with tabs[3]:
    full_metric_map = artifact_metric_lookup(full_artifact)
    rows = []
    for target in [PRIMARY_TARGET, SECONDARY_TARGET]:
        low_best_row = best_row(best_models_df, selected_subset_name, target)
        full_row = full_metric_map.get(target, {})
        full_rmse = full_row.get("RMSE")
        low_rmse = float(low_best_row["best_rmse"])
        rows.append(
            {
                "Target": TARGET_LABELS[target],
                "Full-data model": full_artifact_model_name(full_artifact, full_row),
                "Full-data RMSE": full_rmse,
                f"Low-data ({selected_subset}) model": low_best_row["best_model"],
                f"Low-data ({selected_subset}) RMSE": low_rmse,
                "RMSE change": None if full_rmse is None else low_rmse - float(full_rmse),
            }
        )
    if full_pilstm is not None and low_pilstm is not None:
        rows.append(
            {
                "Target": "PI-LSTM hot outlet",
                "Full-data model": "PI-LSTM",
                "Full-data RMSE": float(full_pilstm["metrics"]["hot_outlet"]["RMSE"]),
                f"Low-data ({selected_subset}) model": "PI-LSTM",
                f"Low-data ({selected_subset}) RMSE": float(low_pilstm["metrics"]["hot_outlet"]["RMSE"]),
                "RMSE change": float(low_pilstm["metrics"]["hot_outlet"]["RMSE"]) - float(full_pilstm["metrics"]["hot_outlet"]["RMSE"]),
            }
        )
    comparison_df = pd.DataFrame(rows)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    change_chart = px.bar(comparison_df.dropna(subset=["RMSE change"]), x="Target", y="RMSE change", color="Target", title=f"RMSE change from full data to the {selected_subset}-row subset")
    st.plotly_chart(change_chart, use_container_width=True)

with tabs[4]:
    hybrid_hot = results["predicted_hot_outlet_k_hybrid"]
    st.markdown(
        dedent(
            f"""
            - PI-LSTM is included here as the sequence-aware, physics-informed comparison model for hot outlet prediction.
            - Version 2 keeps that comparison inside the same digital-twin story, but retrains the workflow on low data.
            - On the current scenario, the low-data traditional hot-outlet model gives {format_temperature(results['predicted_hot_outlet_k_ml'])} and the low-data hybrid layer gives {format_temperature(hybrid_hot)}.
            - PI-LSTM gives {pilstm_hot} when the low-data PI-LSTM artifact is available.
            - The dashboard now focuses on the PI-LSTM evidence instead of separate MLP/VanillaLSTM fields.
            """
        ).strip()
    )
