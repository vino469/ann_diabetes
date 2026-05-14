# ANN Diabetes Prediction Project

This project demonstrates how to build an Artificial Neural Network (ANN) using TensorFlow and Keras to predict whether a person has diabetes based on medical diagnostic measurements.

---

# 📌 Project Overview

The project includes:

- Data Loading
- Data Preprocessing
- Feature Scaling
- Train-Test Splitting
- Building ANN Model
- Model Training
- Model Evaluation
- Prediction Generation

---

# 🧠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Keras

---

# 📂 Dataset Information

The project uses the **Pima Indians Diabetes Dataset**.

## Features

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

## Target

- `0` → No Diabetes
- `1` → Diabetes

---

# ⚙️ Data Preprocessing

Feature scaling is performed using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
