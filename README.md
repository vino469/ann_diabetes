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
🧠 ANN Model Architecture

The ANN model consists of:

Input Layer with 8 features
Hidden Layer 1 → 12 neurons with ReLU activation
Hidden Layer 2 → 8 neurons with ReLU activation
Output Layer → 1 neuron with Sigmoid activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(12, input_dim=8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
⚡ Model Compilation

The model uses:

Optimizer → Adam
Loss Function → Binary Crossentropy
Metric → Accuracy
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
🚀 Model Training

The model is trained for 100 epochs with a batch size of 10.

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=10
)
📊 Model Evaluation

The model performance is evaluated using test data.

loss, accuracy = model.evaluate(X_test, y_test)

print("Loss:", loss)
print("Accuracy:", accuracy)
Evaluation Metrics
Accuracy
Precision
Recall
F1-Score
Confusion Matrix

Example:

from sklearn.metrics import classification_report, confusion_matrix

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob >= 0.5)

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

ann_diabetes_project - task.ipynb
📁 Project Structure
├── diabetes.csv
├── ann_diabetes_project - task.ipynb
├── README.md
🎯 Conclusion

This project demonstrates how Deep Learning can be applied to medical datasets for binary classification problems such as diabetes prediction. Using an ANN with TensorFlow and Keras helps achieve effective prediction performance with structured healthcare data.
