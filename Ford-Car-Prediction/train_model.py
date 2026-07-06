import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

# Load dataset
from pathlib import Path

BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "ford.csv")

# Features and Target
X = df.drop("price", axis=1)
y = df["price"]

# Categorical and Numerical Columns
categorical_features = ["model", "transmission", "fuelType"]
numerical_features = ["year", "mileage", "tax", "mpg", "engineSize"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ]
)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
pipeline.fit(X_train, y_train)

# Save Model
joblib.dump(pipeline, "model.pkl")

print("Model trained and saved successfully!")