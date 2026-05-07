# Heat Exchanger Digital Twin with PI-LSTM

## Quick Start

Run these commands from the repository root:

```bash
# Run the Version 1 dashboard
streamlit run version_1/streamlit_app.py

# Train the standard Version 1 models and save the model artifact
python train_model.py
```

## Files

**Main Application:**
- `streamlit_app.py` - Dashboard for the standard Version 1 workflow
- `heat_exchanger_best_model.py` - ML utilities

**Training Scripts:**
- `train_best_heat_exchanger_artifact.py` - Train traditional ML
- `train_model.py` - Dedicated standard training script that saves the prediction artifact

**Data & Models:**
- `heat_exchanger_dataset.csv` - Dataset
- `best_heat_exchanger_models.joblib` - Saved trained artifact for Version 1

**Notebook:**
- `heat_exchanger_research_from_scratch_colab.ipynb` - Complete analysis

## Models

| Model | Hot Outlet Accuracy |
|-------|-------------------|
| Traditional ML | ~98% |
| Hybrid | ~98.5% |

## Dashboard Features

- Standard ML and hybrid prediction cards only
- Dynamic accuracy display
- Realistic predictions
- Professional styling
- Data quality warning

## Requirements

```bash
pip install streamlit pandas numpy plotly joblib scikit-learn
```

> TensorFlow is optional and only needed for version-2 PI-LSTM artifacts or legacy PI-LSTM training.
