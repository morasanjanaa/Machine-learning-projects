"""
Configuration file for Heart Disease Prediction Project.
"""

# Dataset
DATA_PATH = "data/heart.csv"

# Saved Model Paths
MODEL_PATH = "models/heart_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Training Parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Target Column
TARGET_COLUMN = "HeartDisease"