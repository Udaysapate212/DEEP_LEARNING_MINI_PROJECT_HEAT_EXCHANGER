from __future__ import annotations

from pathlib import Path
import argparse
import joblib
from pprint import pprint

from version_1.heat_exchanger_best_model import FLUID_PRESETS, predict_scenario

ROOT_DIR = Path(__file__).resolve().parent
VERSION1_DIR = ROOT_DIR / "version_1"
ARTIFACT_PATH = VERSION1_DIR / "best_heat_exchanger_models.joblib"


def load_artifact(path: Path | str = ARTIFACT_PATH) -> dict:
    import sys
    from pathlib import Path as PathLib

    ROOT_DIR = PathLib(__file__).resolve().parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {artifact_path}. Run `python train_model.py` first."
        )
    return joblib.load(artifact_path)


def get_fluid_properties(name: str) -> dict[str, float]:
    if name == "Custom":
        return FLUID_PRESETS["Water"]
    return FLUID_PRESETS.get(name, FLUID_PRESETS["Water"])


def predict(
    hot_inlet_temperature_k: float,
    cold_inlet_temperature_k: float,
    cold_inlet_mass_flow_kg_s: float,
    hot_inlet_temperature_k_noisy: float,
    cold_inlet_mass_flow_kg_s_noisy: float,
    hot_fluid_name: str,
    cold_fluid_name: str,
) -> dict[str, float]:
    artifact = load_artifact()
    hot_props = get_fluid_properties(hot_fluid_name)
    cold_props = get_fluid_properties(cold_fluid_name)

    return predict_scenario(
        artifact=artifact,
        hot_inlet_temperature_k=hot_inlet_temperature_k,
        hot_inlet_temperature_k_noisy=hot_inlet_temperature_k_noisy,
        cold_inlet_temperature_k=cold_inlet_temperature_k,
        cold_inlet_mass_flow_kg_s=cold_inlet_mass_flow_kg_s,
        cold_inlet_mass_flow_kg_s_noisy=cold_inlet_mass_flow_kg_s_noisy,
        hot_props=hot_props,
        cold_props=cold_props,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the saved Version 1 artifact and predict heat exchanger outputs from live input values."
    )
    parser.add_argument("--hot-inlet-temperature-k", type=float, default=500.0)
    parser.add_argument("--cold-inlet-temperature-k", type=float, default=293.15)
    parser.add_argument("--cold-inlet-mass-flow-kg-s", type=float, default=2.5)
    parser.add_argument("--hot-sensor-bias-k", type=float, default=0.0)
    parser.add_argument("--cold-flow-sensor-bias-kg-s", type=float, default=0.0)
    parser.add_argument("--hot-fluid", type=str, default="Water", choices=[*FLUID_PRESETS.keys(), "Custom"])
    parser.add_argument("--cold-fluid", type=str, default="Water", choices=[*FLUID_PRESETS.keys(), "Custom"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hot_noisy = args.hot_inlet_temperature_k + args.hot_sensor_bias_k
    cold_flow_noisy = args.cold_inlet_mass_flow_kg_s + args.cold_flow_sensor_bias_kg_s

    predictions = predict(
        hot_inlet_temperature_k=args.hot_inlet_temperature_k,
        cold_inlet_temperature_k=args.cold_inlet_temperature_k,
        cold_inlet_mass_flow_kg_s=args.cold_inlet_mass_flow_kg_s,
        hot_inlet_temperature_k_noisy=hot_noisy,
        cold_inlet_mass_flow_kg_s_noisy=cold_flow_noisy,
        hot_fluid_name=args.hot_fluid,
        cold_fluid_name=args.cold_fluid,
    )

    print("Loaded artifact:", ARTIFACT_PATH)
    print("Input values:")
    pprint({
        "hot_inlet_temperature_k": args.hot_inlet_temperature_k,
        "hot_inlet_temperature_k_noisy": hot_noisy,
        "cold_inlet_temperature_k": args.cold_inlet_temperature_k,
        "cold_inlet_mass_flow_kg_s": args.cold_inlet_mass_flow_kg_s,
        "cold_inlet_mass_flow_kg_s_noisy": cold_flow_noisy,
        "hot_fluid": args.hot_fluid,
        "cold_fluid": args.cold_fluid,
    })
    print("Predictions:")
    pprint(predictions)


if __name__ == "__main__":
    main()
