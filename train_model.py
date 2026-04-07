"""
Training script for Physics-Informed LSTM Heat Exchanger Model
This script trains the model from scratch using only input features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from physics_informed_lstm import PhysicsInformedLSTM
from sklearn.model_selection import train_test_split

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('heat_exchanger_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Initialize the model
print("\nInitializing Physics-Informed LSTM model...")
pi_lstm = PhysicsInformedLSTM(
    sequence_length=10,
    lstm_units=64,
    learning_rate=0.001
)

# Prepare data (using only input features, ignoring existing outputs)
print("\nPreparing data...")
X, y = pi_lstm.prepare_data(df)
print(f"Input shape: {X.shape}")
print(f"Output shape: {y.shape}")

# Create sequences for LSTM
X_seq, y_seq = pi_lstm.create_sequences(X, y)
print(f"Sequence input shape: {X_seq.shape}")
print(f"Sequence output shape: {y_seq.shape}")

# Split data: 70% train, 15% validation, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X_seq, y_seq, test_size=0.15, random_state=42, shuffle=False
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, shuffle=False  # 0.176 * 0.85 ≈ 0.15
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Build the model
print("\nBuilding model architecture...")
model = pi_lstm.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
model.summary()

# Train the model
print("\n" + "="*50)
print("Starting training...")
print("="*50)
history = pi_lstm.train(
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    batch_size=32
)

# Evaluate on test set
print("\n" + "="*50)
print("Evaluating on test set...")
print("="*50)
results, y_pred = pi_lstm.evaluate(X_test, y_test)

print("\nTest Set Performance:")
print("-" * 50)
print("Hot Outlet Temperature:")
print(f"  MAE:  {results['hot_outlet']['MAE']:.4f} K")
print(f"  RMSE: {results['hot_outlet']['RMSE']:.4f} K")
print(f"  MAPE: {results['hot_outlet']['MAPE']:.4f} %")
print("\nCold Outlet Temperature:")
print(f"  MAE:  {results['cold_outlet']['MAE']:.4f} K")
print(f"  RMSE: {results['cold_outlet']['RMSE']:.4f} K")
print(f"  MAPE: {results['cold_outlet']['MAPE']:.4f} %")

# Save the model
print("\nSaving model...")
pi_lstm.model.save('physics_informed_lstm_model.h5')
print("Model saved as 'physics_informed_lstm_model.h5'")

# Plot training history
print("\nGenerating training plots...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Model Loss During Training')
axes[0].legend()
axes[0].grid(True)

# MAE plot
axes[1].plot(history.history['mae'], label='Training MAE')
axes[1].plot(history.history['val_mae'], label='Validation MAE')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].set_title('Mean Absolute Error During Training')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("Training history saved as 'training_history.png'")

# Plot predictions vs actual
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Hot outlet temperature
axes[0].scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5, s=10)
axes[0].plot([y_test[:, 0].min(), y_test[:, 0].max()], 
             [y_test[:, 0].min(), y_test[:, 0].max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Hot Outlet Temperature (K)')
axes[0].set_ylabel('Predicted Hot Outlet Temperature (K)')
axes[0].set_title('Hot Outlet Temperature: Predicted vs Actual')
axes[0].legend()
axes[0].grid(True)

# Cold outlet temperature
axes[1].scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5, s=10)
axes[1].plot([y_test[:, 1].min(), y_test[:, 1].max()], 
             [y_test[:, 1].min(), y_test[:, 1].max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Cold Outlet Temperature (K)')
axes[1].set_ylabel('Predicted Cold Outlet Temperature (K)')
axes[1].set_title('Cold Outlet Temperature: Predicted vs Actual')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
print("Predictions plot saved as 'predictions_vs_actual.png'")

# Save predictions to CSV for comparison
comparison_df = pd.DataFrame({
    'actual_hot_outlet': y_test[:, 0],
    'predicted_hot_outlet': y_pred[:, 0],
    'error_hot_outlet': y_test[:, 0] - y_pred[:, 0],
    'actual_cold_outlet': y_test[:, 1],
    'predicted_cold_outlet': y_pred[:, 1],
    'error_cold_outlet': y_test[:, 1] - y_pred[:, 1]
})
comparison_df.to_csv('predictions_comparison.csv', index=False)
print("Predictions saved as 'predictions_comparison.csv'")

print("\n" + "="*50)
print("Training and evaluation complete!")
print("="*50)
