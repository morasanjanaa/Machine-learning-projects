import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import joblib


st.set_page_config(
    page_title="Insurance Charges Prediction",
    page_icon="🏥",
    layout="centered"
)

model = joblib.load("Insurance-Prediction/model.pkl")

st.title("🏥 Insurance Charges Prediction")

st.write("Enter the details below.")

age = st.number_input("Age", 18, 100, 25)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    10.0,
    60.0,
    25.0
)

children = st.number_input(
    "Children",
    0,
    10,
    0
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    [
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    ]
)

if st.button("Predict Insurance Charges"):

    input_data = pd.DataFrame({
        "age":[age],
        "sex":[sex],
        "bmi":[bmi],
        "children":[children],
        "smoker":[smoker],
        "region":[region]
    })

    prediction = model.predict(input_data)

    st.success(f"Estimated Insurance Charge: ${prediction[0]:,.2f}")
