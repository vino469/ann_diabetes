# ANN Diabetes Prediction Project

This project demonstrates how to build an Artificial Neural Network (ANN) using TensorFlow and Keras to predict whether a patient has diabetes based on medical diagnostic measurements.

---

## 📌 Project Overview

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

## 🧠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Keras

---

## 📂 Dataset Information

The project uses the **Pima Indians Diabetes Dataset**.

### Features

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

### Target

- `0` → No Diabetes
- `1` → Diabetes

---

## ⚙️ Data Preprocessing

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
✂️ Train-Test Splitting
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
🧠 ANN Model Architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(12, input_dim=8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
⚡ Model Compilation
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
🚀 Model Training
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=10,
    validation_split=0.2
)
📊 Model Evaluation
loss, accuracy = model.evaluate(X_test, y_test)

print("Loss:", loss)
print("Accuracy:", accuracy)

The project evaluates the model using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype(int)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
▶️ How to Run the Project
Step 1: Clone Repository
git clone your-repository-link
Step 2: Install Required Libraries
pip install pandas numpy scikit-learn tensorflow
Step 3: Run Jupyter Notebook
jupyter notebook

Then open:

ann_diabetes_project_task.ipynb
📁 Project Structure
├── diabetes.csv
├── ann_diabetes_project_task.ipynb
├── README.md
🎯 Future Improvements
Add Dropout Layers
Hyperparameter Tuning
Deploy Using Streamlit
Improve Accuracy
Add Visualization Dashboard
Conclusion

This project provides a beginner-friendly implementation of an Artificial Neural Network for diabetes prediction using TensorFlow and Keras. It demonstrates how deep learning can be applied to healthcare-related prediction problems using structured medical data.
