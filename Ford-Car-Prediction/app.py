import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Ford Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
BASE_DIR = Path(__file__).parent
model = joblib.load(BASE_DIR / "model.pkl")

# ----------------------------
# Title
# ----------------------------
st.title("🚗 Ford Car Price Prediction")

st.write("Enter the vehicle details below to predict the estimated selling price.")

# ----------------------------
# User Inputs
# ----------------------------

car_model = st.selectbox(
    "Car Model",
    [
        " Fiesta",
        " Focus",
        " Focus Estate",
        " Fusion",
        " B-MAX",
        " C-MAX",
        " Grand C-MAX",
        " Kuga",
        " EcoSport",
        " Mondeo",
        " Ka+",
        " S-MAX",
        " Galaxy",
        " Edge",
        " Mustang",
        " Puma",
        " Tourneo Custom"
    ]
)

year = st.number_input(
    "Manufacturing Year",
    min_value=1996,
    max_value=2024,
    value=2018
)

transmission = st.selectbox(
    "Transmission",
    [
        "Manual",
        "Automatic",
        "Semi-Auto"
    ]
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    value=25000
)

fuelType = st.selectbox(
    "Fuel Type",
    [
        "Petrol",
        "Diesel",
        "Hybrid",
        "Electric",
        "Other"
    ]
)

tax = st.number_input(
    "Road Tax (£)",
    min_value=0,
    value=145
)

mpg = st.number_input(
    "Miles Per Gallon (MPG)",
    min_value=0.0,
    value=55.4
)

engineSize = st.number_input(
    "Engine Size (L)",
    min_value=0.0,
    value=1.5
)

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "model": [car_model],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuelType],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engineSize]
    })

    prediction = model.predict(input_df)

    st.success(f"💰 Estimated Car Price: £{prediction[0]:,.2f}")