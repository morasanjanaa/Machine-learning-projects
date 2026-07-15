import streamlit as st
from predict import predict

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- CSS ---------------- #

st.markdown("""
<style>

.stApp{
    background-color:#f5f7fa;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#D7263D;
}

.subtitle{
    text-align:center;
    color:#666666;
    font-size:18px;
    margin-bottom:30px;
}

.block{
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 3px 12px rgba(0,0,0,.12);
}

.result{
    text-align:center;
    font-size:32px;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #

st.markdown("<div class='title'>❤️ Heart Disease Prediction</div>", unsafe_allow_html=True)

st.markdown(
"<div class='subtitle'>Predict the likelihood of heart disease using Machine Learning.</div>",
unsafe_allow_html=True)

# ---------------- INPUTS ---------------- #

st.markdown("<div class='block'>", unsafe_allow_html=True)

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input("Age", 20, 100, 45)

    sex = st.selectbox(
        "Sex",
        ["M","F"]
    )

    chest = st.selectbox(
        "Chest Pain Type",
        ["ATA","NAP","ASY","TA"]
    )

    bp = st.number_input(
        "Resting Blood Pressure",
        80,
        220,
        120
    )

    chol = st.number_input(
        "Cholesterol",
        0,
        700,
        220
    )

with col2:

    fasting = st.selectbox(
        "Fasting Blood Sugar",
        [0,1]
    )

    ecg = st.selectbox(
        "Resting ECG",
        ["Normal","ST","LVH"]
    )

    hr = st.number_input(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    angina = st.selectbox(
        "Exercise Angina",
        ["N","Y"]
    )

    oldpeak = st.number_input(
        "Old Peak",
        0.0,
        10.0,
        1.0
    )

slope = st.selectbox(
    "ST Slope",
    ["Up","Flat","Down"]
)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- BUTTON ---------------- #

st.write("")

predict_btn = st.button(
    "🔍 Predict Heart Disease",
    use_container_width=True
)

# ---------------- PREDICTION ---------------- #

if predict_btn:

    patient = {

        "Age": age,
        "Sex": sex,
        "ChestPainType": chest,
        "RestingBP": bp,
        "Cholesterol": chol,
        "FastingBS": fasting,
        "RestingECG": ecg,
        "MaxHR": hr,
        "ExerciseAngina": angina,
        "Oldpeak": oldpeak,
        "ST_Slope": slope

    }

    prediction, confidence = predict(patient)

    st.write("")

    st.markdown("<div class='block'>", unsafe_allow_html=True)

    st.subheader("Prediction")

    if prediction == 1:

        st.error("❤️ High Risk of Heart Disease")

    else:

        st.success("💚 Low Risk of Heart Disease")

    st.progress(float(confidence))

    st.metric(
        "Prediction Confidence",
        f"{confidence*100:.2f}%"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Model","KNN")

    c2.metric("Accuracy","89.13%")

    c3.metric("Features","11")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #

st.write("")

st.markdown("---")

st.markdown(
"""
<div class='footer'>
Made with ❤️ using Streamlit & Scikit-Learn
</div>
""",
unsafe_allow_html=True
)