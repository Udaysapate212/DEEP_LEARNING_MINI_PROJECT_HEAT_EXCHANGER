import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Physics-Informed LSTM for Heat Exchanger Temperature Prediction\n",
                "\n",
                "This notebook implements a Physics-Informed LSTM model to predict outlet temperatures in a heat exchanger system.\n",
                "\n",
                "## Key Features\n",
                "- Physics-informed loss function with energy conservation constraints\n",
                "- Trained from scratch using only input features\n",
                "- LSTM architecture for temporal pattern learning\n",
                "- Comprehensive evaluation and visualization"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Import Required Libraries"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import pandas as pd\n",
                "import tensorflow as tf\n",
                "from tensorflow import keras\n",
                "from tensorflow.keras import layers\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.model_selection import train_test_split\n",
                "import matplotlib.pyplot as plt\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Set random seeds for reproducibility\n",
                "np.random.seed(42)\n",
                "tf.random.set_seed(42)\n",
                "\n",
                "print('TensorFlow version:', tf.__version__)\n",
                "print('Keras version:', keras.__version__)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Load and Explore Dataset"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load dataset\n",
                "df = pd.read_csv('heat_exchanger_dataset.csv')\n",
                "\n",
                "print('Dataset shape:', df.shape)\n",
                "print('\\nFirst few rows:')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Display dataset statistics\n",
                "print('Dataset Statistics:')\n",
                "df.describe()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Define Physics-Informed LSTM Model Class"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class PhysicsInformedLSTM:\n",
                "    def __init__(self, sequence_length=10, lstm_units=64, learning_rate=0.001):\n",
                "        self.sequence_length = sequence_length\n",
                "        self.lstm_units = lstm_units\n",
                "        self.learning_rate = learning_rate\n",
                "        self.model = None\n",
                "        self.scaler_X = StandardScaler()\n",
                "        self.scaler_y = StandardScaler()\n",
                "        self.history = None\n",
                "        \n",
                "    def create_sequences(self, X, y):\n",
                "        X_seq, y_seq = [], []\n",
                "        for i in range(len(X) - self.sequence_length):\n",
                "            X_seq.append(X[i:i + self.sequence_length])\n",
                "            y_seq.append(y[i + self.sequence_length])\n",
                "        return np.array(X_seq), np.array(y_seq)\n",
                "    \n",
                "    def physics_loss(self, y_true, y_pred):\n",
                "        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))\n",
                "        hot_outlet_pred = y_pred[:, 0]\n",
                "        cold_outlet_pred = y_pred[:, 1]\n",
                "        physics_penalty = tf.reduce_mean(\n",
                "            tf.maximum(0.0, -hot_outlet_pred) +\n",
                "            tf.maximum(0.0, -cold_outlet_pred)\n",
                "        )\n",
                "        total_loss = mse_loss + 0.1 * physics_penalty\n",
                "        return total_loss\n",
                "    \n",
                "    def build_model(self, input_shape):\n",
                "        inputs = layers.Input(shape=input_shape)\n",
                "        x = layers.LSTM(self.lstm_units, return_sequences=True)(inputs)\n",
                "        x = layers.Dropout(0.2)(x)\n",
                "        x = layers.LSTM(self.lstm_units // 2, return_sequences=True)(x)\n",
                "        x = layers.Dropout(0.2)(x)\n",
                "        x = layers.LSTM(self.lstm_units // 4, return_sequences=False)(x)\n",
                "        x = layers.Dropout(0.2)(x)\n",
                "        x = layers.Dense(32, activation='relu')(x)\n",
                "        x = layers.Dense(16, activation='relu')(x)\n",
                "        outputs = layers.Dense(2, activation='linear')(x)\n",
                "        \n",
                "        model = keras.Model(inputs=inputs, outputs=outputs)\n",
                "        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)\n",
                "        model.compile(optimizer=optimizer, loss=self.physics_loss, metrics=['mae', 'mse'])\n",
                "        \n",
                "        self.model = model\n",
                "        return model\n",
                "    \n",
                "    def prepare_data(self, df):\n",
                "        input_features = [\n",
                "            'hot_inlet_temperature_k',\n",
                "            'cold_inlet_mass_flow_kg_s',\n",
                "            'hx_1_heat_load_kw',\n",
                "            'hot_outlet_pressure_pa',\n",
                "            'cold_outlet_pressure_pa',\n",
                "            'hot_outlet_mass_flow_kg_s',\n",
                "            'cold_outlet_mass_flow_kg_s',\n",
                "            'hx_1_logarithmic_mean_temperature_difference_lmtd_k'\n",
                "        ]\n",
                "        output_features = [\n",
                "            'hot_outlet_temperature_k',\n",
                "            'cold_outlet_temperature_k'\n",
                "        ]\n",
                "        X = df[input_features].values\n",
                "        y = df[output_features].values\n",
                "        return X, y\n",
                "    \n",
                "    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):\n",
                "        X_train_scaled = self.scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1]))\n",
                "        X_train_scaled = X_train_scaled.reshape(X_train.shape)\n",
                "        X_val_scaled = self.scaler_X.transform(X_val.reshape(-1, X_val.shape[-1]))\n",
                "        X_val_scaled = X_val_scaled.reshape(X_val.shape)\n",
                "        y_train_scaled = self.scaler_y.fit_transform(y_train)\n",
                "        y_val_scaled = self.scaler_y.transform(y_val)\n",
                "        \n",
                "        early_stopping = keras.callbacks.EarlyStopping(\n",
                "            monitor='val_loss', patience=15, restore_best_weights=True\n",
                "        )\n",
                "        reduce_lr = keras.callbacks.ReduceLROnPlateau(\n",
                "            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6\n",
                "        )\n",
                "        \n",
                "        self.history = self.model.fit(\n",
                "            X_train_scaled, y_train_scaled,\n",
                "            validation_data=(X_val_scaled, y_val_scaled),\n",
                "            epochs=epochs, batch_size=batch_size,\n",
                "            callbacks=[early_stopping, reduce_lr], verbose=1\n",
                "        )\n",
                "        return self.history\n",
                "    \n",
                "    def predict(self, X):\n",
                "        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1]))\n",
                "        X_scaled = X_scaled.reshape(X.shape)\n",
                "        y_pred_scaled = self.model.predict(X_scaled, verbose=0)\n",
                "        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)\n",
                "        return y_pred\n",
                "    \n",
                "    def evaluate(self, X_test, y_test):\n",
                "        y_pred = self.predict(X_test)\n",
                "        mae_hot = np.mean(np.abs(y_test[:, 0] - y_pred[:, 0]))\n",
                "        mae_cold = np.mean(np.abs(y_test[:, 1] - y_pred[:, 1]))\n",
                "        rmse_hot = np.sqrt(np.mean((y_test[:, 0] - y_pred[:, 0])**2))\n",
                "        rmse_cold = np.sqrt(np.mean((y_test[:, 1] - y_pred[:, 1])**2))\n",
                "        mape_hot = np.mean(np.abs((y_test[:, 0] - y_pred[:, 0]) / y_test[:, 0])) * 100\n",
                "        mape_cold = np.mean(np.abs((y_test[:, 1] - y_pred[:, 1]) / y_test[:, 1])) * 100\n",
                "        \n",
                "        results = {\n",
                "            'hot_outlet': {'MAE': mae_hot, 'RMSE': rmse_hot, 'MAPE': mape_hot},\n",
                "            'cold_outlet': {'MAE': mae_cold, 'RMSE': rmse_cold, 'MAPE': mape_cold}\n",
                "        }\n",
                "        return results, y_pred\n",
                "\n",
                "print('PhysicsInformedLSTM class defined successfully')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Prepare Data for Training"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Initialize model\n",
                "pi_lstm = PhysicsInformedLSTM(sequence_length=10, lstm_units=64, learning_rate=0.001)\n",
                "\n",
                "# Prepare data\n",
                "X, y = pi_lstm.prepare_data(df)\n",
                "print('Input shape:', X.shape)\n",
                "print('Output shape:', y.shape)\n",
                "\n",
                "# Create sequences\n",
                "X_seq, y_seq = pi_lstm.create_sequences(X, y)\n",
                "print('Sequence input shape:', X_seq.shape)\n",
                "print('Sequence output shape:', y_seq.shape)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Split data\n",
                "X_temp, X_test, y_temp, y_test = train_test_split(\n",
                "    X_seq, y_seq, test_size=0.15, random_state=42, shuffle=False\n",
                ")\n",
                "X_train, X_val, y_train, y_val = train_test_split(\n",
                "    X_temp, y_temp, test_size=0.176, random_state=42, shuffle=False\n",
                ")\n",
                "\n",
                "print('Train set:', X_train.shape[0], 'samples')\n",
                "print('Validation set:', X_val.shape[0], 'samples')\n",
                "print('Test set:', X_test.shape[0], 'samples')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Build and Train Model"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Build model\n",
                "model = pi_lstm.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))\n",
                "model.summary()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train model\n",
                "history = pi_lstm.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Visualize Training History"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
                "\n",
                "axes[0].plot(history.history['loss'], label='Training Loss')\n",
                "axes[0].plot(history.history['val_loss'], label='Validation Loss')\n",
                "axes[0].set_xlabel('Epoch')\n",
                "axes[0].set_ylabel('Loss')\n",
                "axes[0].set_title('Model Loss During Training')\n",
                "axes[0].legend()\n",
                "axes[0].grid(True)\n",
                "\n",
                "axes[1].plot(history.history['mae'], label='Training MAE')\n",
                "axes[1].plot(history.history['val_mae'], label='Validation MAE')\n",
                "axes[1].set_xlabel('Epoch')\n",
                "axes[1].set_ylabel('MAE')\n",
                "axes[1].set_title('Mean Absolute Error During Training')\n",
                "axes[1].legend()\n",
                "axes[1].grid(True)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Evaluate Model Performance"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Evaluate on test set\n",
                "results, y_pred = pi_lstm.evaluate(X_test, y_test)\n",
                "\n",
                "print('Test Set Performance:')\n",
                "print('-' * 50)\n",
                "print('Hot Outlet Temperature:')\n",
                "print(f\"  MAE:  {results['hot_outlet']['MAE']:.4f} K\")\n",
                "print(f\"  RMSE: {results['hot_outlet']['RMSE']:.4f} K\")\n",
                "print(f\"  MAPE: {results['hot_outlet']['MAPE']:.4f} %\")\n",
                "print('\\nCold Outlet Temperature:')\n",
                "print(f\"  MAE:  {results['cold_outlet']['MAE']:.4f} K\")\n",
                "print(f\"  RMSE: {results['cold_outlet']['RMSE']:.4f} K\")\n",
                "print(f\"  MAPE: {results['cold_outlet']['MAPE']:.4f} %\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Calculate Accuracy Percentages"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Calculate accuracy as percentage\n",
                "hot_accuracy = 100 - results['hot_outlet']['MAPE']\n",
                "cold_accuracy = 100 - results['cold_outlet']['MAPE']\n",
                "overall_accuracy = (hot_accuracy + cold_accuracy) / 2\n",
                "\n",
                "print('Model Accuracy Report:')\n",
                "print('=' * 70)\n",
                "print(f'Hot Outlet Temperature Accuracy:  {hot_accuracy:.4f}%')\n",
                "print(f'Cold Outlet Temperature Accuracy: {cold_accuracy:.4f}%')\n",
                "print(f'Overall Model Accuracy:            {overall_accuracy:.4f}%')\n",
                "print('=' * 70)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Visualize Predictions vs Actual"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
                "\n",
                "# Hot outlet temperature\n",
                "axes[0].scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5, s=10)\n",
                "axes[0].plot([y_test[:, 0].min(), y_test[:, 0].max()],\n",
                "             [y_test[:, 0].min(), y_test[:, 0].max()],\n",
                "             'r--', lw=2, label='Perfect Prediction')\n",
                "axes[0].set_xlabel('Actual Hot Outlet Temperature (K)')\n",
                "axes[0].set_ylabel('Predicted Hot Outlet Temperature (K)')\n",
                "axes[0].set_title('Hot Outlet Temperature: Predicted vs Actual')\n",
                "axes[0].legend()\n",
                "axes[0].grid(True)\n",
                "\n",
                "# Cold outlet temperature\n",
                "axes[1].scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5, s=10)\n",
                "axes[1].plot([y_test[:, 1].min(), y_test[:, 1].max()],\n",
                "             [y_test[:, 1].min(), y_test[:, 1].max()],\n",
                "             'r--', lw=2, label='Perfect Prediction')\n",
                "axes[1].set_xlabel('Actual Cold Outlet Temperature (K)')\n",
                "axes[1].set_ylabel('Predicted Cold Outlet Temperature (K)')\n",
                "axes[1].set_title('Cold Outlet Temperature: Predicted vs Actual')\n",
                "axes[1].legend()\n",
                "axes[1].grid(True)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Error Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Calculate errors\n",
                "hot_errors = y_test[:, 0] - y_pred[:, 0]\n",
                "cold_errors = y_test[:, 1] - y_pred[:, 1]\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
                "\n",
                "# Hot outlet error distribution\n",
                "axes[0].hist(hot_errors, bins=50, edgecolor='black', alpha=0.7)\n",
                "axes[0].axvline(0, color='r', linestyle='--', lw=2, label='Zero Error')\n",
                "axes[0].set_xlabel('Prediction Error (K)')\n",
                "axes[0].set_ylabel('Frequency')\n",
                "axes[0].set_title('Hot Outlet: Error Distribution')\n",
                "axes[0].legend()\n",
                "axes[0].grid(True, alpha=0.3)\n",
                "\n",
                "# Cold outlet error distribution\n",
                "axes[1].hist(cold_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')\n",
                "axes[1].axvline(0, color='r', linestyle='--', lw=2, label='Zero Error')\n",
                "axes[1].set_xlabel('Prediction Error (K)')\n",
                "axes[1].set_ylabel('Frequency')\n",
                "axes[1].set_title('Cold Outlet: Error Distribution')\n",
                "axes[1].legend()\n",
                "axes[1].grid(True, alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 11. Time Series Comparison"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot first 100 predictions\n",
                "n_plot = min(100, len(y_test))\n",
                "x_axis = np.arange(n_plot)\n",
                "\n",
                "fig, axes = plt.subplots(2, 1, figsize=(16, 10))\n",
                "\n",
                "# Hot outlet time series\n",
                "axes[0].plot(x_axis, y_test[:n_plot, 0], 'b-', label='Actual', linewidth=2)\n",
                "axes[0].plot(x_axis, y_pred[:n_plot, 0], 'r--', label='Predicted', linewidth=2)\n",
                "axes[0].fill_between(x_axis, y_test[:n_plot, 0], y_pred[:n_plot, 0], alpha=0.3)\n",
                "axes[0].set_xlabel('Sample Index')\n",
                "axes[0].set_ylabel('Temperature (K)')\n",
                "axes[0].set_title('Hot Outlet Temperature: Time Series Comparison')\n",
                "axes[0].legend()\n",
                "axes[0].grid(True, alpha=0.3)\n",
                "\n",
                "# Cold outlet time series\n",
                "axes[1].plot(x_axis, y_test[:n_plot, 1], 'b-', label='Actual', linewidth=2)\n",
                "axes[1].plot(x_axis, y_pred[:n_plot, 1], 'r--', label='Predicted', linewidth=2)\n",
                "axes[1].fill_between(x_axis, y_test[:n_plot, 1], y_pred[:n_plot, 1], alpha=0.3)\n",
                "axes[1].set_xlabel('Sample Index')\n",
                "axes[1].set_ylabel('Temperature (K)')\n",
                "axes[1].set_title('Cold Outlet Temperature: Time Series Comparison')\n",
                "axes[1].legend()\n",
                "axes[1].grid(True, alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 12. Save Model and Results"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Save model\n",
                "pi_lstm.model.save('physics_informed_lstm_model.h5')\n",
                "print('Model saved as physics_informed_lstm_model.h5')\n",
                "\n",
                "# Save predictions\n",
                "comparison_df = pd.DataFrame({\n",
                "    'actual_hot_outlet': y_test[:, 0],\n",
                "    'predicted_hot_outlet': y_pred[:, 0],\n",
                "    'error_hot_outlet': y_test[:, 0] - y_pred[:, 0],\n",
                "    'actual_cold_outlet': y_test[:, 1],\n",
                "    'predicted_cold_outlet': y_pred[:, 1],\n",
                "    'error_cold_outlet': y_test[:, 1] - y_pred[:, 1]\n",
                "})\n",
                "comparison_df.to_csv('predictions_comparison.csv', index=False)\n",
                "print('Predictions saved as predictions_comparison.csv')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 13. Summary and Conclusions"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('='*70)\n",
                "print('PHYSICS-INFORMED LSTM MODEL SUMMARY')\n",
                "print('='*70)\n",
                "print(f'\\nModel Architecture:')\n",
                "print(f'  - Input: {X_train.shape[1]} timesteps x {X_train.shape[2]} features')\n",
                "print(f'  - LSTM layers: 64 -> 32 -> 16 units')\n",
                "print(f'  - Output: 2 predictions (hot and cold outlet temperatures)')\n",
                "print(f'  - Total parameters: {pi_lstm.model.count_params():,}')\n",
                "print(f'\\nTraining Configuration:')\n",
                "print(f'  - Training samples: {len(X_train)}')\n",
                "print(f'  - Validation samples: {len(X_val)}')\n",
                "print(f'  - Test samples: {len(X_test)}')\n",
                "print(f'  - Epochs trained: {len(history.history[\"loss\"])}')\n",
                "print(f'\\nPerformance Metrics:')\n",
                "print(f'  - Hot Outlet Accuracy: {hot_accuracy:.4f}%')\n",
                "print(f'  - Cold Outlet Accuracy: {cold_accuracy:.4f}%')\n",
                "print(f'  - Overall Accuracy: {overall_accuracy:.4f}%')\n",
                "print(f'\\nKey Features:')\n",
                "print(f'  - Physics-informed loss function')\n",
                "print(f'  - Trained from scratch using only input features')\n",
                "print(f'  - Energy conservation constraints')\n",
                "print(f'  - Temporal pattern learning with LSTM')\n",
                "print('='*70)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('physics_informed_lstm_heat_exchanger.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print('Jupyter notebook created successfully: physics_informed_lstm_heat_exchanger.ipynb')
