"""
Physics-Informed LSTM Model for Heat Exchanger Temperature Prediction
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


def physics_loss(y_true, y_pred):
    """Custom physics-informed loss function"""
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    hot_outlet_pred = y_pred[:, 0]
    cold_outlet_pred = y_pred[:, 1]
    physics_penalty = tf.reduce_mean(
        tf.maximum(0.0, -hot_outlet_pred) +
        tf.maximum(0.0, -cold_outlet_pred)
    )
    total_loss = mse_loss + 0.1 * physics_penalty
    return total_loss


class PhysicsInformedLSTM:
    def __init__(self, sequence_length=10, lstm_units=64, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.history = None
        
    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        x = layers.LSTM(self.lstm_units, return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(self.lstm_units // 2, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(self.lstm_units // 4, return_sequences=False)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(16, activation='relu')(x)
        outputs = layers.Dense(2, activation='linear')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=physics_loss, metrics=['mae', 'mse'])
        self.model = model
        return model
    
    def prepare_data(self, df_in):
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
        output_features = [
            'hot_outlet_temperature_k',
            'cold_outlet_temperature_k'
        ]
        X = df_in[input_features].values
        y = df_in[output_features].values
        return X, y
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=1):
        X_train_scaled = self.scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_val_scaled = self.scaler_X.transform(X_val.reshape(-1, X_val.shape[-1]))
        X_val_scaled = X_val_scaled.reshape(X_val.shape)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        
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
        
        self.history = self.model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        return self.history
    
    def predict(self, X):
        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1]))
        X_scaled = X_scaled.reshape(X.shape)
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred
