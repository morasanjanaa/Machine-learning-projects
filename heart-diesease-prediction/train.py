"""
Train Heart Disease Prediction Model using a Pipeline
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score

from src.config import DATA_PATH, TARGET_COLUMN


# ------------------------
# Load Dataset
# ------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COLUMN])

y = df[TARGET_COLUMN]


# ------------------------
# Split
# ------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------------
# Columns
# ------------------------

categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()

numeric_columns = X.select_dtypes(exclude=["object"]).columns.tolist()


# ------------------------
# Preprocessor
# ------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_columns
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)


# ------------------------
# Models
# ------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "KNN": KNeighborsClassifier(),

    "Naive Bayes": GaussianNB(),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Support Vector Machine": SVC(probability=True)
}


best_model = None
best_accuracy = 0
best_name = ""


print("=" * 60)
print("Training Models")
print("=" * 60)

for name, classifier in models.items():

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    f1 = f1_score(y_test, pred)

    print(f"{name}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("-" * 40)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name


os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/heart_model.pkl")

print("\nBest Model :", best_name)
print("Accuracy :", best_accuracy)
print("\nModel Saved Successfully!")