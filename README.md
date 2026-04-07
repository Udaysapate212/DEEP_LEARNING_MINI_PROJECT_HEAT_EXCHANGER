# Physics-Informed LSTM for Heat Exchanger Temperature Prediction

This project implements a Physics-Informed Long Short-Term Memory (LSTM) neural network to predict outlet temperatures (hot and cold) in a heat exchanger system.

## Key Features

- **Physics-Informed Architecture**: Incorporates physical constraints and energy conservation principles
- **Custom Loss Function**: Combines MSE with physics-based penalties for physically impossible predictions
- **From-Scratch Training**: Model is trained only on input features, completely ignoring pre-existing output values in the dataset
- **Sequence-Based Learning**: Uses LSTM to capture temporal dependencies in heat exchanger operation

## Model Architecture

The model consists of:
- 3 stacked LSTM layers (64 → 32 → 16 units) with dropout regularization
- 2 dense layers for feature extraction
- 2 output neurons for hot and cold outlet temperatures
- Custom physics-informed loss function

## Input Features

The model uses the following input features:
1. Hot inlet temperature (K)
2. Cold inlet mass flow rate (kg/s)
3. Heat load (kW)
4. Hot outlet pressure (Pa)
5. Cold outlet pressure (Pa)
6. Hot outlet mass flow rate (kg/s)
7. Cold outlet mass flow rate (kg/s)
8. Logarithmic Mean Temperature Difference - LMTD (K)

## Output Predictions

The model predicts:
1. Hot outlet temperature (K)
2. Cold outlet temperature (K)

## Physics Constraints

The model incorporates:
- Energy conservation principles
- Temperature bounds (preventing negative temperatures)
- Heat transfer relationships
- Mass flow conservation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python train_model.py
```

This will:
- Load the dataset
- Prepare sequences for LSTM
- Train the model from scratch
- Evaluate performance on test set
- Generate visualization plots
- Save the trained model and predictions

### Output Files

After training, you'll get:
- `physics_informed_lstm_model.h5` - Trained model
- `training_history.png` - Loss and MAE curves
- `predictions_vs_actual.png` - Scatter plots comparing predictions with actual values
- `predictions_comparison.csv` - Detailed comparison of predictions vs actual values

## Model Performance

The model is evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **MAPE** (Mean Absolute Percentage Error)

Performance metrics are calculated separately for hot and cold outlet temperatures.

## Project Structure

```
.
├── heat_exchanger_dataset.csv          # Input dataset
├── physics_informed_lstm.py            # Model class definition
├── train_model.py                      # Training script
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Important Notes

- The model is trained from scratch using ONLY input features
- Pre-existing output values in the dataset are completely ignored during training
- The model learns the underlying physics and relationships independently
- Sequence length is set to 10 time steps for temporal pattern learning
- Early stopping and learning rate reduction are used to prevent overfitting

## Customization

You can modify hyperparameters in `train_model.py`:
- `sequence_length`: Number of time steps in each sequence (default: 10)
- `lstm_units`: Number of units in LSTM layers (default: 64)
- `learning_rate`: Initial learning rate (default: 0.001)
- `epochs`: Maximum training epochs (default: 100)
- `batch_size`: Training batch size (default: 32)

## Physics-Informed Loss Function

The custom loss function combines:
1. Standard MSE loss for prediction accuracy
2. Physics penalty terms for constraint violations
3. Weighted combination to balance accuracy and physical validity

This ensures predictions are both accurate and physically meaningful.
