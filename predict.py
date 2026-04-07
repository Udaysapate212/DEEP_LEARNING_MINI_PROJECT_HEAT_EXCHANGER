"""
Prediction script for trained Physics-Informed LSTM model
Use this to make predictions on new data
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pickle

def load_model_and_scalers():
    """Load the trained model and scalers"""
    model = keras.models.load_model(
        'physics_informed_lstm_model.h5',
        custom_objects={'physics_loss': lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred))}
    )
    return model

def prepare_input_data(data, sequence_length=10):
    """
    Prepare input data for prediction
    
    Parameters:
    -----------
    data : pandas DataFrame or numpy array
        Input features in the correct order:
        [hot_inlet_temperature_k, cold_inlet_mass_flow_kg_s, hx_1_heat_load_kw,
         hot_outlet_pressure_pa, cold_outlet_pressure_pa, hot_outlet_mass_flow_kg_s,
         cold_outlet_mass_flow_kg_s, hx_1_logarithmic_mean_temperature_difference_lmtd_k]
    sequence_length : int
        Number of time steps in sequence
    
    Returns:
    --------
    sequences : numpy array
        Prepared sequences for LSTM input
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # Create sequences
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    
    return np.array(sequences)

def predict_temperatures(model, input_data, scaler_X, scaler_y):
    """
    Make temperature predictions
    
    Parameters:
    -----------
    model : keras Model
        Trained LSTM model
    input_data : numpy array
        Prepared input sequences
    scaler_X : StandardScaler
        Fitted scaler for input features
    scaler_y : StandardScaler
        Fitted scaler for output features
    
    Returns:
    --------
    predictions : numpy array
        Predicted [hot_outlet_temp, cold_outlet_temp]
    """
    # Scale input
    input_scaled = scaler_X.transform(input_data.reshape(-1, input_data.shape[-1]))
    input_scaled = input_scaled.reshape(input_data.shape)
    
    # Predict
    predictions_scaled = model.predict(input_scaled)
    
    # Inverse transform
    predictions = scaler_y.inverse_transform(predictions_scaled)
    
    return predictions

# Example usage
if __name__ == "__main__":
    print("Loading trained model...")
    model = load_model_and_scalers()
    
    print("\nModel loaded successfully!")
    print("Model expects input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)
    
    print("\n" + "="*60)
    print("To use this model for predictions:")
    print("="*60)
    print("\n1. Prepare your input data with these features (in order):")
    print("   - hot_inlet_temperature_k")
    print("   - cold_inlet_mass_flow_kg_s")
    print("   - hx_1_heat_load_kw")
    print("   - hot_outlet_pressure_pa")
    print("   - cold_outlet_pressure_pa")
    print("   - hot_outlet_mass_flow_kg_s")
    print("   - cold_outlet_mass_flow_kg_s")
    print("   - hx_1_logarithmic_mean_temperature_difference_lmtd_k")
    print("\n2. Create sequences of length 10")
    print("\n3. Use the predict_temperatures() function")
    print("\nExample:")
    print("  predictions = predict_temperatures(model, input_sequences, scaler_X, scaler_y)")
    print("  hot_outlet = predictions[:, 0]")
    print("  cold_outlet = predictions[:, 1]")
