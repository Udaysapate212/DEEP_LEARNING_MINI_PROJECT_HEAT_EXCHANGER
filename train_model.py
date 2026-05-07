from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from version_1.heat_exchanger_best_model import (
    DEFAULT_CONFIG,
    FLUID_PRESETS,
    train_best_linear_model,
)

ROOT_DIR = Path(__file__).resolve().parent
VERSION1_DIR = ROOT_DIR / "version_1"
DATASET_PATH = VERSION1_DIR / "heat_exchanger_dataset.csv"
ARTIFACT_PATH = VERSION1_DIR / "best_heat_exchanger_models.joblib"
METRICS_PATH = VERSION1_DIR / "best_model_metrics.csv"
HEAT_LOAD_PREDICTIONS_PATH = VERSION1_DIR / "best_model_heat_load_predictions.csv"
HOT_OUTLET_PREDICTIONS_PATH = VERSION1_DIR / "best_model_hot_outlet_predictions.csv"
COLD_OUTLET_PREDICTIONS_PATH = VERSION1_DIR / "cold_outlet_predictions.csv"


def build_artifact(
    heat_load_model, hot_outlet_model, cold_outlet_model, metrics_df
) -> dict[str, object]:
    return {
        "artifact_name": "Heat Exchanger Best Models",
        "best_model_family": "LinearRegressionGD",
        "selection_basis": (
            "Chosen from the shared Version 1 model flow where standard tabular models "
            "are trained on the full dataset and saved as a reusable artifact."
        ),
        "config": dict(DEFAULT_CONFIG),
        "fluid_presets": FLUID_PRESETS,
        "models": {
            heat_load_model.target_name: heat_load_model,
            hot_outlet_model.target_name: hot_outlet_model,
            cold_outlet_model.target_name: cold_outlet_model,
        },
        "metrics": metrics_df.to_dict(orient="records"),
    }


def main() -> None:
    dataset_path = DATASET_PATH
    print(f"Loading dataset from {dataset_path}")

    heat_load_model, heat_load_predictions = train_best_linear_model(
        dataset_path=str(dataset_path),
        target_name="hx_1_heat_load_kw",
        config=DEFAULT_CONFIG,
    )
    hot_outlet_model, hot_outlet_predictions = train_best_linear_model(
        dataset_path=str(dataset_path),
        target_name="hot_outlet_temperature_k",
        config=DEFAULT_CONFIG,
    )
    cold_outlet_model, cold_outlet_predictions = train_best_linear_model(
        dataset_path=str(dataset_path),
        target_name="cold_outlet_temperature_k",
        config=DEFAULT_CONFIG,
    )

    metrics_df = pd.DataFrame(
        [
            {"Target": heat_load_model.target_name, **heat_load_model.metrics},
            {"Target": hot_outlet_model.target_name, **hot_outlet_model.metrics},
            {"Target": cold_outlet_model.target_name, **cold_outlet_model.metrics},
        ]
    )

    artifact = build_artifact(
        heat_load_model=heat_load_model,
        hot_outlet_model=hot_outlet_model,
        cold_outlet_model=cold_outlet_model,
        metrics_df=metrics_df,
    )

    joblib.dump(artifact, ARTIFACT_PATH)
    metrics_df.to_csv(METRICS_PATH, index=False)
    heat_load_predictions.to_csv(HEAT_LOAD_PREDICTIONS_PATH, index=False)
    hot_outlet_predictions.to_csv(HOT_OUTLET_PREDICTIONS_PATH, index=False)
    cold_outlet_predictions.to_csv(COLD_OUTLET_PREDICTIONS_PATH, index=False)

    print(f"Saved trained artifact: {ARTIFACT_PATH.resolve()}")
    print(f"Saved metrics: {METRICS_PATH.resolve()}")
    print(f"Saved heat-load predictions: {HEAT_LOAD_PREDICTIONS_PATH.resolve()}")
    print(f"Saved hot-outlet predictions: {HOT_OUTLET_PREDICTIONS_PATH.resolve()}")
    print(f"Saved cold-outlet predictions: {COLD_OUTLET_PREDICTIONS_PATH.resolve()}")
    print("\nTraining complete. The app will only load this artifact for runtime prediction.")


if __name__ == "__main__":
    main()
