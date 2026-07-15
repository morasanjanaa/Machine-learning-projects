"""
Utility functions.
"""

import joblib


def save_model(model, path):
    """
    Save trained model.
    """
    joblib.dump(model, path)


def load_model(path):
    """
    Load trained model.
    """
    return joblib.load(path)