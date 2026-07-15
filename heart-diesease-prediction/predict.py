import pandas as pd
import joblib

model = joblib.load("models/heart_model.pkl")


def predict(features):

    df = pd.DataFrame([features])

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0]

    confidence = probability.max()

    return prediction, confidence