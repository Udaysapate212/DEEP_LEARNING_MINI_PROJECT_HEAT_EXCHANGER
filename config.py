"""
Configuration file for Physics-Informed LSTM model
Modify these parameters to customize the model behavior
"""

# Model Architecture Parameters
MODEL_CONFIG = {
    'sequence_length': 10,      # Number of time steps in each sequence
    'lstm_units': 64,           # Number of units in first LSTM layer
    'learning_rate': 0.001,     # Initial learning rate for Adam optimizer
    'dropout_rate': 0.2,        # Dropout rate for regularization
}

# Training Parameters
TRAINING_CONFIG = {
    'epochs': 100,              # Maximum number of training epochs
    'batch_size': 32,           # Training batch size
    'validation_split': 0.15,   # Fraction of data for validation
    'test_split': 0.15,         # Fraction of data for testing
    'early_stopping_patience': 15,  # Epochs to wait before early stopping
    'reduce_lr_patience': 5,    # Epochs to wait before reducing learning rate
    'reduce_lr_factor': 0.5,    # Factor to reduce learning rate
    'min_lr': 1e-6,            # Minimum learning rate
}

# Physics Loss Parameters
PHYSICS_CONFIG = {
    'physics_penalty_weight': 0.1,  # Weight for physics penalty in loss function
    'enable_temperature_bounds': True,  # Enable temperature bound constraints
    'enable_energy_conservation': True,  # Enable energy conservation constraints
}

# Data Parameters
DATA_CONFIG = {
    'dataset_path': 'heat_exchanger_dataset.csv',
    'input_features': [
        'hot_inlet_temperature_k',
        'cold_inlet_mass_flow_kg_s',
        'hx_1_heat_load_kw',
        'hot_outlet_pressure_pa',
        'cold_outlet_pressure_pa',
        'hot_outlet_mass_flow_kg_s',
        'cold_outlet_mass_flow_kg_s',
        'hx_1_logarithmic_mean_temperature_difference_lmtd_k'
    ],
    'output_features': [
        'hot_outlet_temperature_k',
        'cold_outlet_temperature_k'
    ],
    'shuffle_data': False,  # Whether to shuffle data before splitting
    'random_seed': 42,      # Random seed for reproducibility
}

# Visualization Parameters
VIZ_CONFIG = {
    'plot_dpi': 300,           # DPI for saved plots
    'plot_style': 'default',   # Matplotlib style
    'figure_size': (15, 5),    # Default figure size
    'save_plots': True,        # Whether to save plots
    'show_plots': False,       # Whether to display plots (set False for headless)
}

# Model Saving Parameters
SAVE_CONFIG = {
    'model_path': 'physics_informed_lstm_model.h5',
    'save_best_only': True,    # Save only the best model during training
    'save_weights_only': False, # Save full model or just weights
}

# Prediction Parameters
PREDICT_CONFIG = {
    'prediction_batch_size': 64,  # Batch size for predictions
    'verbose': 1,                 # Verbosity level (0, 1, or 2)
}

# Advanced Options
ADVANCED_CONFIG = {
    'use_mixed_precision': False,  # Use mixed precision training (requires GPU)
    'use_tensorboard': False,      # Enable TensorBoard logging
    'tensorboard_log_dir': './logs',  # TensorBoard log directory
    'profile_training': False,     # Profile training performance
}

# Hyperparameter Tuning Suggestions
TUNING_SUGGESTIONS = {
    'sequence_length': [5, 10, 15, 20],
    'lstm_units': [32, 64, 128, 256],
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64, 128],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4],
}

def print_config():
    """Print current configuration"""
    print("="*70)
    print("CURRENT CONFIGURATION")
    print("="*70)
    
    print("\nModel Architecture:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nTraining Parameters:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nPhysics Constraints:")
    for key, value in PHYSICS_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nData Configuration:")
    print(f"  dataset_path: {DATA_CONFIG['dataset_path']}")
    print(f"  num_input_features: {len(DATA_CONFIG['input_features'])}")
    print(f"  num_output_features: {len(DATA_CONFIG['output_features'])}")
    print(f"  shuffle_data: {DATA_CONFIG['shuffle_data']}")
    print(f"  random_seed: {DATA_CONFIG['random_seed']}")
    
    print("\nVisualization:")
    for key, value in VIZ_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    print_config()
