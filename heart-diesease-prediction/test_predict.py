from predict import predict

sample = {
    "Age": 40,
    "Sex": "M",
    "ChestPainType": "ATA",
    "RestingBP": 120,
    "Cholesterol": 230,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 170,
    "ExerciseAngina": "N",
    "Oldpeak": 0.0,
    "ST_Slope": "Up"
}

prediction, confidence = predict(sample)

print(prediction)
print(confidence)