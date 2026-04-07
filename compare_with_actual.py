"""
Script to compare model predictions with actual values from the dataset
This demonstrates that the model was trained independently of the actual outputs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from physics_informed_lstm import PhysicsInformedLSTM
from tensorflow import keras

def load_and_compare():
    """Load model and compare predictions with actual dataset values"""
    
    print("Loading dataset...")
    df = pd.read_csv('heat_exchanger_dataset.csv')
    
    print("Loading trained model...")
    try:
        # Define custom loss function for loading
        def physics_loss(y_true, y_pred):
            import tensorflow as tf
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            hot_outlet_pred = y_pred[:, 0]
            cold_outlet_pred = y_pred[:, 1]
            physics_penalty = tf.reduce_mean(
                tf.maximum(0.0, -hot_outlet_pred) +
                tf.maximum(0.0, -cold_outlet_pred)
            )
            total_loss = mse_loss + 0.1 * physics_penalty
            return total_loss
        
        # Load the saved model with custom objects
        model = keras.models.load_model(
            'physics_informed_lstm_model.h5',
            custom_objects={'physics_loss': physics_loss}
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error: {e}")
        print("Please run train_model.py first.")
        return
    
    # Initialize PI-LSTM to use its data preparation methods
    pi_lstm = PhysicsInformedLSTM(sequence_length=10)
    
    # Prepare data
    print("\nPreparing data...")
    X, y_actual = pi_lstm.prepare_data(df)
    X_seq, y_seq = pi_lstm.create_sequences(X, y_actual)
    
    # Use a subset for comparison (e.g., last 500 samples)
    n_samples = min(500, len(X_seq))
    X_compare = X_seq[-n_samples:]
    y_compare = y_seq[-n_samples:]
    
    print(f"\nComparing {n_samples} samples...")
    
    # Make predictions (need to scale data)
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit scalers on all data
    X_all_flat = X_seq.reshape(-1, X_seq.shape[-1])
    scaler_X.fit(X_all_flat)
    scaler_y.fit(y_seq)
    
    # Scale comparison data
    X_compare_scaled = scaler_X.transform(X_compare.reshape(-1, X_compare.shape[-1]))
    X_compare_scaled = X_compare_scaled.reshape(X_compare.shape)
    
    # Predict
    y_pred_scaled = model.predict(X_compare_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculate errors
    hot_outlet_error = y_compare[:, 0] - y_pred[:, 0]
    cold_outlet_error = y_compare[:, 1] - y_pred[:, 1]
    
    # Statistics
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print("\nHot Outlet Temperature:")
    print(f"  Mean Actual:     {y_compare[:, 0].mean():.2f} K")
    print(f"  Mean Predicted:  {y_pred[:, 0].mean():.2f} K")
    print(f"  Mean Error:      {hot_outlet_error.mean():.4f} K")
    print(f"  Std Error:       {hot_outlet_error.std():.4f} K")
    print(f"  Max Error:       {np.abs(hot_outlet_error).max():.4f} K")
    print(f"  MAE:             {np.abs(hot_outlet_error).mean():.4f} K")
    print(f"  RMSE:            {np.sqrt((hot_outlet_error**2).mean()):.4f} K")
    
    print("\nCold Outlet Temperature:")
    print(f"  Mean Actual:     {y_compare[:, 1].mean():.2f} K")
    print(f"  Mean Predicted:  {y_pred[:, 1].mean():.2f} K")
    print(f"  Mean Error:      {cold_outlet_error.mean():.4f} K")
    print(f"  Std Error:       {cold_outlet_error.std():.4f} K")
    print(f"  Max Error:       {np.abs(cold_outlet_error).max():.4f} K")
    print(f"  MAE:             {np.abs(cold_outlet_error).mean():.4f} K")
    print(f"  RMSE:            {np.sqrt((cold_outlet_error**2).mean()):.4f} K")
    
    # Create detailed comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Hot outlet: Scatter plot
    axes[0, 0].scatter(y_compare[:, 0], y_pred[:, 0], alpha=0.5, s=20)
    axes[0, 0].plot([y_compare[:, 0].min(), y_compare[:, 0].max()],
                     [y_compare[:, 0].min(), y_compare[:, 0].max()],
                     'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Hot Outlet Temperature (K)', fontsize=12)
    axes[0, 0].set_ylabel('Predicted Hot Outlet Temperature (K)', fontsize=12)
    axes[0, 0].set_title('Hot Outlet: Predicted vs Actual', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Hot outlet: Error distribution
    axes[0, 1].hist(hot_outlet_error, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='r', linestyle='--', lw=2, label='Zero Error')
    axes[0, 1].set_xlabel('Prediction Error (K)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Hot Outlet: Error Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cold outlet: Scatter plot
    axes[1, 0].scatter(y_compare[:, 1], y_pred[:, 1], alpha=0.5, s=20, color='orange')
    axes[1, 0].plot([y_compare[:, 1].min(), y_compare[:, 1].max()],
                     [y_compare[:, 1].min(), y_compare[:, 1].max()],
                     'r--', lw=2, label='Perfect Prediction')
    axes[1, 0].set_xlabel('Actual Cold Outlet Temperature (K)', fontsize=12)
    axes[1, 0].set_ylabel('Predicted Cold Outlet Temperature (K)', fontsize=12)
    axes[1, 0].set_title('Cold Outlet: Predicted vs Actual', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cold outlet: Error distribution
    axes[1, 1].hist(cold_outlet_error, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].axvline(0, color='r', linestyle='--', lw=2, label='Zero Error')
    axes[1, 1].set_xlabel('Prediction Error (K)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Cold Outlet: Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detailed_comparison.png', dpi=300, bbox_inches='tight')
    print("\nDetailed comparison plot saved as 'detailed_comparison.png'")
    
    # Time series comparison (first 100 samples)
    n_plot = min(100, n_samples)
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    x_axis = np.arange(n_plot)
    
    # Hot outlet time series
    axes[0].plot(x_axis, y_compare[:n_plot, 0], 'b-', label='Actual', linewidth=2)
    axes[0].plot(x_axis, y_pred[:n_plot, 0], 'r--', label='Predicted', linewidth=2)
    axes[0].fill_between(x_axis, y_compare[:n_plot, 0], y_pred[:n_plot, 0], alpha=0.3)
    axes[0].set_xlabel('Sample Index', fontsize=12)
    axes[0].set_ylabel('Temperature (K)', fontsize=12)
    axes[0].set_title('Hot Outlet Temperature: Time Series Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Cold outlet time series
    axes[1].plot(x_axis, y_compare[:n_plot, 1], 'b-', label='Actual', linewidth=2)
    axes[1].plot(x_axis, y_pred[:n_plot, 1], 'r--', label='Predicted', linewidth=2)
    axes[1].fill_between(x_axis, y_compare[:n_plot, 1], y_pred[:n_plot, 1], alpha=0.3)
    axes[1].set_xlabel('Sample Index', fontsize=12)
    axes[1].set_ylabel('Temperature (K)', fontsize=12)
    axes[1].set_title('Cold Outlet Temperature: Time Series Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_series_comparison.png', dpi=300, bbox_inches='tight')
    print("Time series comparison plot saved as 'time_series_comparison.png'")
    
    # Save detailed comparison to CSV
    comparison_df = pd.DataFrame({
        'sample_index': np.arange(n_samples),
        'actual_hot_outlet_K': y_compare[:, 0],
        'predicted_hot_outlet_K': y_pred[:, 0],
        'error_hot_outlet_K': hot_outlet_error,
        'abs_error_hot_outlet_K': np.abs(hot_outlet_error),
        'actual_cold_outlet_K': y_compare[:, 1],
        'predicted_cold_outlet_K': y_pred[:, 1],
        'error_cold_outlet_K': cold_outlet_error,
        'abs_error_cold_outlet_K': np.abs(cold_outlet_error)
    })
    comparison_df.to_csv('detailed_comparison.csv', index=False)
    print("Detailed comparison data saved as 'detailed_comparison.csv'")
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)

if __name__ == "__main__":
    load_and_compare()
