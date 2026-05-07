from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from heat_exchanger_best_model import FLUID_PRESETS, predict_scenario

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = APP_DIR / "best_heat_exchanger_models.joblib"

# This app only performs inference from a saved Version 1 artifact.
# The artifact is produced offline by `python train_model.py`.
st.set_page_config(
    page_title="Version 1 Heat Exchanger Twin",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_artifact() -> dict:
    import sys
    from pathlib import Path

    ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    if not ARTIFACT_PATH.exists():
        st.error(
            f"Saved model artifact not found at {ARTIFACT_PATH}. Run `python train_model.py` first."
        )
        st.stop()
    return joblib.load(ARTIFACT_PATH)


def inject_styles() -> None:
    st.markdown(
        dedent(
            """
            <style>
            .stApp { background: linear-gradient(135deg, #f0f4f8 0%, #ffffff 100%); }
            .stMarkdown, .main .block-container, .main .block-container * { color: #183642 !important; }
            .hero-card { background: rgba(255,255,255,0.92); border: 1px solid rgba(24,54,66,0.12); border-radius: 18px; padding: 1.4rem; box-shadow: 0 18px 40px rgba(0,0,0,0.06); color: #183642; }
            .prediction-card { background: rgba(255,255,255,0.96); border: 1px solid rgba(24,54,66,0.12); border-radius: 18px; padding: 1.1rem; color: #183642; }
            .prediction-label { font-size: 0.9rem; font-weight: 700; color: #1f3f49; margin-bottom: 0.5rem; }
            .prediction-value { font-size: 1.8rem; font-weight: 800; color: #0f4c75; }
            .prediction-note { color: #3a5f72; font-size: 0.92rem; margin-top: 0.5rem; }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )


def fluid_block(title: str, prefix: str) -> dict[str, float]:
    preset_name = st.selectbox(
        f"{title} preset",
        [*FLUID_PRESETS.keys(), "Custom"],
        key=f"{prefix}_preset",
    )
    preset = FLUID_PRESETS.get(preset_name, FLUID_PRESETS["Water"])
    c1, c2 = st.columns(2)
    with c1:
        cp = st.number_input(
            f"{title} Cp (kJ/kgK)",
            min_value=0.1,
            max_value=10.0,
            value=float(preset["cp_kj_kgk"]),
            step=0.01,
            key=f"{prefix}_cp",
        )
        rho = st.number_input(
            f"{title} density (kg/m3)",
            min_value=1.0,
            max_value=2500.0,
            value=float(preset["rho_kg_m3"]),
            step=1.0,
            key=f"{prefix}_rho",
        )
    with c2:
        mu = st.number_input(
            f"{title} viscosity (Pa·s)",
            min_value=0.00001,
            max_value=5.0,
            value=float(preset["mu_pa_s"]),
            step=0.0001,
            format="%.5f",
            key=f"{prefix}_mu",
        )
        k = st.number_input(
            f"{title} conductivity (W/mK)",
            min_value=0.01,
            max_value=5.0,
            value=float(preset["k_w_mk"]),
            step=0.01,
            key=f"{prefix}_k",
        )
    return {"cp_kj_kgk": cp, "rho_kg_m3": rho, "mu_pa_s": mu, "k_w_mk": k}


def format_temperature(value_k: float) -> str:
    return f"{value_k:.3f} K / {value_k - 273.15:.3f} °C"


def render_prediction_card(column, label: str, value: str, note: str) -> None:
    column.markdown(
        dedent(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">{label}</div>
                <div class="prediction-value">{value}</div>
                <div class="prediction-note">{note}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    st.title("Version 1: Standard Heat Exchanger Prediction")
    st.write(
        "This app loads a pre-trained standard ML artifact and performs inference on current user inputs. "
        "No training happens during runtime."
    )

    artifact = load_artifact()

    with st.sidebar:
        st.header("Input Scenario")
        selected_hot_fluid = st.selectbox("Hot fluid", [*FLUID_PRESETS.keys()], index=0)
        selected_cold_fluid = st.selectbox("Cold fluid", [*FLUID_PRESETS.keys()], index=0)
        hot_inlet_temperature_k = st.slider(
            "Hot inlet temperature (K)", 320.0, 650.0, 473.15, step=1.0
        )
        cold_inlet_temperature_k = st.slider(
            "Cold inlet temperature (K)", 280.0, 350.0, 293.15, step=1.0)
        cold_inlet_mass_flow_kg_s = st.slider(
            "Cold inlet mass flow (kg/s)", 0.30, 6.00, 2.75, step=0.01)
        hot_sensor_bias_k = st.slider(
            "Hot sensor bias (K)", -20.0, 20.0, 0.0, step=0.1)
        cold_flow_sensor_bias_kg_s = st.slider(
            "Cold flow sensor bias (kg/s)", -0.50, 0.50, 0.0, step=0.01)
        run_prediction = st.button("Run prediction")

    st.markdown("---")
    st.subheader("Prediction workflow")
    st.markdown(
        "1. Load the saved Version 1 artifact from disk.\n"
        "2. Build the input feature row from the current sidebar values.\n"
        "3. Use the saved scalers and trained models only for inference.\n"
        "4. Present the standard ML and hybrid outputs without any PI-LSTM comparison."
    )

    if not run_prediction:
        st.info("Adjust the scenario inputs on the left and click \"Run prediction\" to see output.")
        st.stop()

    hot_props = fluid_block("Hot fluid properties", "hot")
    cold_props = fluid_block("Cold fluid properties", "cold")

    results = predict_scenario(
        artifact=artifact,
        hot_inlet_temperature_k=hot_inlet_temperature_k,
        hot_inlet_temperature_k_noisy=hot_inlet_temperature_k + hot_sensor_bias_k,
        cold_inlet_temperature_k=cold_inlet_temperature_k,
        cold_inlet_mass_flow_kg_s=cold_inlet_mass_flow_kg_s,
        cold_inlet_mass_flow_kg_s_noisy=cold_inlet_mass_flow_kg_s + cold_flow_sensor_bias_kg_s,
        hot_props=hot_props,
        cold_props=cold_props,
    )

    st.subheader("Current scenario predictions")
    c1, c2, c3 = st.columns(3)
    render_prediction_card(
        c1,
        "Standard ML heat load",
        f"{results['predicted_heat_load_kw_ml']:.2f} kW",
        "Saved trained tabular model prediction using current sidebar inputs.",
    )
    render_prediction_card(
        c2,
        "Standard ML hot outlet",
        format_temperature(results["predicted_hot_outlet_k_ml"]),
        "Hot outlet prediction from the saved hot outlet regression model.",
    )
    render_prediction_card(
        c3,
        "Standard ML cold outlet",
        format_temperature(results["predicted_cold_outlet_k_ml"]),
        "Cold outlet prediction from the saved cold outlet regression model.",
    )

    c4, c5, c6 = st.columns(3)
    render_prediction_card(
        c4,
        "Hybrid heat load",
        f"{results['predicted_heat_load_kw_hybrid']:.2f} kW",
        "Hybrid output after fluid-aware adjustment to the ML heat-load result.",
    )
    render_prediction_card(
        c5,
        "Hybrid hot outlet",
        format_temperature(results["predicted_hot_outlet_k_hybrid"]),
        "Hybrid hot outlet after fluid correction.",
    )
    render_prediction_card(
        c6,
        "Hybrid cold outlet",
        format_temperature(results["predicted_cold_outlet_k_hybrid"]),
        "Hybrid cold outlet after fluid correction.",
    )

    st.markdown("---")
    st.subheader("Runtime input summary")
    input_df = pd.DataFrame(
        [
            {
                "Parameter": "Hot inlet temperature (K)",
                "Value": hot_inlet_temperature_k,
            },
            {
                "Parameter": "Hot inlet temperature noisy (K)",
                "Value": hot_inlet_temperature_k + hot_sensor_bias_k,
            },
            {
                "Parameter": "Cold inlet temperature (K)",
                "Value": cold_inlet_temperature_k,
            },
            {
                "Parameter": "Cold inlet mass flow (kg/s)",
                "Value": cold_inlet_mass_flow_kg_s,
            },
            {
                "Parameter": "Cold inlet mass flow noisy (kg/s)",
                "Value": cold_inlet_mass_flow_kg_s + cold_flow_sensor_bias_kg_s,
            },
            {
                "Parameter": "Hot fluid preset", "Value": selected_hot_fluid,
            },
            {
                "Parameter": "Cold fluid preset",
                "Value": selected_cold_fluid,
            },
        ]
    )
    st.dataframe(input_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Saved artifact metrics")
    metrics_df = pd.DataFrame(artifact["metrics"])
    st.dataframe(metrics_df, use_container_width=True)
    st.caption(
        "These metrics were produced during offline training and are separate from the live prediction path."
    )

    fig = px.bar(
        metrics_df,
        x="Target",
        y="RMSE",
        title="Saved RMSE values for Version 1 models",
        text="RMSE",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
