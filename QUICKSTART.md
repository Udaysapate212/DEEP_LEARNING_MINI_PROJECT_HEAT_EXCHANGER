# Quick Start Guide

## Physics-Informed LSTM for Heat Exchanger Temperature Prediction

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

Run the training script to build and train the model from scratch:

```bash
python train_model.py
```

This will:
- Load the heat exchanger dataset
- Create sequences for LSTM training
- Train the Physics-Informed LSTM model (ignoring existing outputs)
- Evaluate performance on test data
- Generate visualization plots
- Save the trained model

**Expected Output:**
- `physics_informed_lstm_model.h5` - Trained model file
- `training_history.png` - Training/validation loss curves
- `predictions_vs_actual.png` - Prediction accuracy plots
- `predictions_comparison.csv` - Detailed predictions

**Training Time:** Approximately 5-15 minutes depending on your hardware

### Step 3: Compare with Actual Values

After training, run the comparison script to see how your model performs against the actual dataset values:

```bash
python compare_with_actual.py
```

This will:
- Load your trained model
- Make predictions on a subset of data
- Compare predictions with actual values from the dataset
- Generate detailed comparison plots and statistics

**Output Files:**
- `detailed_comparison.png` - Scatter plots and error distributions
- `time_series_comparison.png` - Time series visualization
- `detailed_comparison.csv` - Sample-by-sample comparison

### Step 4: Make Predictions (Optional)

Use the prediction script to make predictions on new data:

```bash
python predict.py
```

This script shows you how to:
- Load the trained model
- Prepare input data in the correct format
- Make temperature predictions

## Understanding the Results

### Performance Metrics

The model reports three key metrics:

1. **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
2. **RMSE (Root Mean Square Error)**: Square root of average squared errors (penalizes large errors more)
3. **MAPE (Mean Absolute Percentage Error)**: Average percentage error

Lower values indicate better performance.

### What Makes This Model Special?

1. **Physics-Informed**: Incorporates physical constraints and energy conservation principles
2. **Independent Training**: Trained from scratch using only input features, completely ignoring pre-existing outputs
3. **Temporal Learning**: Uses LSTM to capture time-dependent patterns in heat exchanger operation
4. **Custom Loss Function**: Combines prediction accuracy with physics-based constraints

### Input Features Used

The model learns from these 8 input features:
- Hot inlet temperature (K)
- Cold inlet mass flow rate (kg/s)
- Heat load (kW)
- Hot outlet pressure (Pa)
- Cold outlet pressure (Pa)
- Hot outlet mass flow rate (kg/s)
- Cold outlet mass flow rate (kg/s)
- Logarithmic Mean Temperature Difference - LMTD (K)

### Predictions Made

The model predicts:
- Hot outlet temperature (K)
- Cold outlet temperature (K)

## Troubleshooting

### Issue: "Model file not found"
**Solution:** Run `python train_model.py` first to train and save the model.

### Issue: "Out of memory" during training
**Solution:** Reduce `batch_size` in `train_model.py` (try 16 or 8 instead of 32).

### Issue: Poor performance
**Solution:** Try:
- Increasing `epochs` (e.g., 150 or 200)
- Adjusting `lstm_units` (try 128)
- Modifying `sequence_length` (try 15 or 20)

## Next Steps

1. Experiment with different hyperparameters
2. Try different sequence lengths
3. Modify the physics constraints in the loss function
4. Add more LSTM layers or change architecture
5. Use the model for real-time predictions

## File Structure

```
.
├── heat_exchanger_dataset.csv       # Your dataset
├── physics_informed_lstm.py         # Model class
├── train_model.py                   # Training script
├── compare_with_actual.py           # Comparison script
├── predict.py                       # Prediction utility
├── requirements.txt                 # Dependencies
├── README.md                        # Full documentation
└── QUICKSTART.md                    # This file
```

## Questions?

Check the full README.md for detailed documentation on:
- Model architecture
- Physics constraints
- Customization options
- Advanced usage

Happy modeling! 🚀
