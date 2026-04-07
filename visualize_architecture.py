"""
Visualize the Physics-Informed LSTM model architecture
"""

from physics_informed_lstm import PhysicsInformedLSTM
import numpy as np

# Create model instance
print("Creating Physics-Informed LSTM model...")
pi_lstm = PhysicsInformedLSTM(
    sequence_length=10,
    lstm_units=64,
    learning_rate=0.001
)

# Build model with sample input shape
# Input shape: (sequence_length, num_features)
# We have 8 input features
input_shape = (10, 8)
model = pi_lstm.build_model(input_shape)

print("\n" + "="*70)
print("PHYSICS-INFORMED LSTM MODEL ARCHITECTURE")
print("="*70)

# Display model summary
model.summary()

print("\n" + "="*70)
print("MODEL DETAILS")
print("="*70)

print(f"\nInput Shape: {model.input_shape}")
print(f"  - Sequence Length: 10 time steps")
print(f"  - Features per time step: 8")

print(f"\nOutput Shape: {model.output_shape}")
print(f"  - Predictions: 2 (hot outlet temp, cold outlet temp)")

print("\nTotal Parameters:", model.count_params())

print("\n" + "="*70)
print("LAYER BREAKDOWN")
print("="*70)

for i, layer in enumerate(model.layers):
    print(f"\nLayer {i+1}: {layer.name}")
    print(f"  Type: {layer.__class__.__name__}")
    print(f"  Output Shape: {layer.output_shape}")
    if hasattr(layer, 'units'):
        print(f"  Units: {layer.units}")
    if hasattr(layer, 'rate'):
        print(f"  Dropout Rate: {layer.rate}")
    if hasattr(layer, 'activation'):
        print(f"  Activation: {layer.activation.__name__ if hasattr(layer.activation, '__name__') else layer.activation}")

print("\n" + "="*70)
print("PHYSICS-INFORMED LOSS FUNCTION")
print("="*70)
print("\nThe model uses a custom loss function that combines:")
print("  1. Mean Squared Error (MSE) for prediction accuracy")
print("  2. Physics-based penalty terms for constraint violations")
print("     - Prevents negative temperatures")
print("     - Enforces energy conservation principles")
print("     - Maintains physical feasibility")

print("\n" + "="*70)
print("INPUT FEATURES (8 total)")
print("="*70)
features = [
    "1. Hot inlet temperature (K)",
    "2. Cold inlet mass flow rate (kg/s)",
    "3. Heat load (kW)",
    "4. Hot outlet pressure (Pa)",
    "5. Cold outlet pressure (Pa)",
    "6. Hot outlet mass flow rate (kg/s)",
    "7. Cold outlet mass flow rate (kg/s)",
    "8. Logarithmic Mean Temperature Difference - LMTD (K)"
]
for feature in features:
    print(f"  {feature}")

print("\n" + "="*70)
print("OUTPUT PREDICTIONS (2 total)")
print("="*70)
outputs = [
    "1. Hot outlet temperature (K)",
    "2. Cold outlet temperature (K)"
]
for output in outputs:
    print(f"  {output}")

print("\n" + "="*70)
print("KEY FEATURES")
print("="*70)
print("  ✓ Stacked LSTM layers for temporal pattern learning")
print("  ✓ Dropout regularization to prevent overfitting")
print("  ✓ Dense layers for feature extraction")
print("  ✓ Physics-informed loss function")
print("  ✓ Adam optimizer with learning rate scheduling")
print("  ✓ Early stopping based on validation loss")

print("\n" + "="*70)
print("TRAINING STRATEGY")
print("="*70)
print("  • Data split: 70% train, 15% validation, 15% test")
print("  • Sequence-based learning with 10 time steps")
print("  • Feature scaling using StandardScaler")
print("  • Early stopping with patience=15 epochs")
print("  • Learning rate reduction on plateau")
print("  • Batch size: 32")
print("  • Maximum epochs: 100")

print("\n" + "="*70)

# Try to visualize if keras.utils.plot_model is available
try:
    from tensorflow.keras.utils import plot_model
    print("\nGenerating model architecture diagram...")
    plot_model(
        model,
        to_file='model_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=150
    )
    print("Model architecture diagram saved as 'model_architecture.png'")
except Exception as e:
    print(f"\nNote: Could not generate architecture diagram.")
    print(f"Install graphviz and pydot to enable this feature:")
    print("  pip install pydot graphviz")

print("\n" + "="*70)
print("Architecture visualization complete!")
print("="*70)
