import pandas as pd
import joblib

from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "heart_model.pkl"

model = joblib.load(MODEL_PATH)

def predict(features):
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    confidence = probability.max()
    return prediction, confidence


def predict(features):

    df = pd.DataFrame([features])

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0]

    confidence = probability.max()

    return prediction, confidence
