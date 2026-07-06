# 🏥 Insurance Charges Prediction

A Machine Learning web application that predicts medical insurance charges based on user information such as age, BMI, smoking status, region, and number of children.

The application is built using **Python, Scikit-learn, and Streamlit** and deployed on **Streamlit Community Cloud**.

---

## 🚀 Live Demo

🔗 **Streamlit App:** https://insurance-prediction-project.streamlit.app/


## 📌 Project Overview

Medical insurance charges vary depending on several factors such as age, BMI, smoking habits, and region.

This project builds a Machine Learning model to predict insurance charges using these features and provides an easy-to-use web interface for making predictions.

---

## 📂 Dataset

The project uses the **Insurance Charges Dataset**.

### Features

- Age
- Sex
- BMI
- Children
- Smoker
- Region

### Target

- Insurance Charges

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 📁 Project Structure

```
Insurance-Prediction/
│
├── notebook/
│   └── EDA_Insurance.ipynb
│
├── app.py
├── train_model.py
├── insurance.csv
├── model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Machine Learning Workflow

1. Load Dataset
2. Perform Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Train-Test Split
5. Train Linear Regression Model
6. Save Model using Joblib
7. Build Streamlit Web App
8. Deploy on Streamlit Cloud

---

## 📊 Model Used

- Linear Regression

---

## 💻 Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Machine-learning-projects.git
```

### Navigate to Project

```bash
cd Machine-learning-projects/Insurance-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train_model.py
```

This generates:

```
model.pkl
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🖥️ Application Preview

### Home Page

- Enter user details
- Click **Predict Insurance Charges**
- View the estimated insurance cost

*(Add a screenshot here after deployment.)*

Example:

```
images/app.png
```

---

## 📈 Features

- Simple and clean user interface
- Real-time insurance charge prediction
- Machine Learning model integration
- Beginner-friendly project structure
- Easy deployment using Streamlit

---

## 📚 Learning Outcomes

This project demonstrates:

- Data preprocessing
- Exploratory Data Analysis
- Machine Learning Pipeline
- Linear Regression
- Model Serialization using Joblib
- Streamlit Deployment
- GitHub Project Management

---

## 🔮 Future Improvements

- Compare multiple regression models
- Hyperparameter tuning
- Better UI with custom CSS
- Model performance visualization
- Docker deployment
- Cloud deployment using AWS/Azure

---

## 👩‍💻 Author

**Mora Sanjana**

GitHub: https://github.com/morasanjanaa

---

## ⭐ If you found this project useful, consider giving it a Star!
