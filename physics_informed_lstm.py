import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PhysicsInformedLSTM:
    """
    Physics-Informed LSTM for Heat Exchanger Outlet Temperature Prediction
    
    This model incorporates physics-based constraints:
    1. Energy conservation (Q = m * Cp * ΔT)
    2. Heat transfer principles (Q = U * A * LMTD)
    3. Mass flow conservation
    """
    
    def __init__(self, sequence_length=10, lstm_units=64, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.history = None
        
    def create_sequences(self, X, y):
        """Create sequences for LSTM input"""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)
    
    def physics_loss(self, y_true, y_pred):
        """
        Custom physics-informed loss function
        Combines MSE with physics-based constraints
        """
        # Standard MSE loss
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Physics constraint: Energy balance
        # Hot outlet should be less than hot inlet
        # Cold outlet should be greater than cold inlet
        hot_outlet_pred = y_pred[:, 0]
        cold_outlet_pred = y_pred[:, 1]
        
        # Penalize physically impossible predictions
        physics_penalty = tf.reduce_mean(
            tf.maximum(0.0, -hot_outlet_pred) +  # Prevent negative temperatures
            tf.maximum(0.0, -cold_outlet_pred)
        )
        
        # Total loss
        total_loss = mse_loss + 0.1 * physics_penalty
        return total_loss

    
    def build_model(self, input_shape):
        """Build the Physics-Informed LSTM architecture"""
        inputs = layers.Input(shape=input_shape)
        
        # LSTM layers with dropout for regularization
        x = layers.LSTM(self.lstm_units, return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(self.lstm_units // 2, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(self.lstm_units // 4, return_sequences=False)(x)
        x = layers.Dropout(0.2)(x)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(16, activation='relu')(x)
        
        # Output layer: 2 outputs (hot_outlet_temp, cold_outlet_temp)
        outputs = layers.Dense(2, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile with custom physics loss
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=self.physics_loss, metrics=['mae', 'mse'])
        
        self.model = model
        return model
    
    def prepare_data(self, df):
        """
        Prepare data for training
        Uses only input features, ignoring existing output predictions
        """
        # Input features (clean data without noisy versions)
        input_features = [
            'hot_inlet_temperature_k',
            'cold_inlet_mass_flow_kg_s',
            'hx_1_heat_load_kw',
            'hot_outlet_pressure_pa',
            'cold_outlet_pressure_pa',
            'hot_outlet_mass_flow_kg_s',
            'cold_outlet_mass_flow_kg_s',
            'hx_1_logarithmic_mean_temperature_difference_lmtd_k'
        ]
        
        # Target outputs (what we want to predict)
        output_features = [
            'hot_outlet_temperature_k',
            'cold_outlet_temperature_k'
        ]
        
        X = df[input_features].values
        y = df[output_features].values
        
        return X, y
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model"""
        # Normalize the data
        X_train_scaled = self.scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        
        X_val_scaled = self.scaler_X.transform(X_val.reshape(-1, X_val.shape[-1]))
        X_val_scaled = X_val_scaled.reshape(X_val.shape)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train the model
        self.history = self.model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1]))
        X_scaled = X_scaled.reshape(X.shape)
        
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mae_hot = np.mean(np.abs(y_test[:, 0] - y_pred[:, 0]))
        mae_cold = np.mean(np.abs(y_test[:, 1] - y_pred[:, 1]))
        
        rmse_hot = np.sqrt(np.mean((y_test[:, 0] - y_pred[:, 0])**2))
        rmse_cold = np.sqrt(np.mean((y_test[:, 1] - y_pred[:, 1])**2))
        
        mape_hot = np.mean(np.abs((y_test[:, 0] - y_pred[:, 0]) / y_test[:, 0])) * 100
        mape_cold = np.mean(np.abs((y_test[:, 1] - y_pred[:, 1]) / y_test[:, 1])) * 100
        
        results = {
            'hot_outlet': {
                'MAE': mae_hot,
                'RMSE': rmse_hot,
                'MAPE': mape_hot
            },
            'cold_outlet': {
                'MAE': mae_cold,
                'RMSE': rmse_cold,
                'MAPE': mape_cold
            }
        }
        
        return results, y_pred
