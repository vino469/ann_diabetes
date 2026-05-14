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
- Matplotlib

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

The dataset is preprocessed before training the ANN model.

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
✂️ Train-Test Splitting

The dataset is divided into training and testing sets.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
🧠 ANN Model Architecture

The ANN model contains:

Input Layer
Two Hidden Layers
Output Layer
Model Structure
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(12, input_dim=8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
Activation Functions
ReLU → Used in hidden layers
Sigmoid → Used in output layer for binary classification
⚡ Model Compilation

The model is compiled using:

Optimizer → Adam
Loss Function → Binary Crossentropy
Evaluation Metric → Accuracy
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
🚀 Model Training

The model is trained for 100 epochs with batch size 10.

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=10,
    validation_split=0.2
)
📊 Model Evaluation

The model performance is evaluated on test data.

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
Evaluation Metrics

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
📈 Accuracy and Loss Visualization
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.show()
🔮 Prediction Example
sample = [[2,120,70,20,79,25.0,0.5,33]]

sample = scaler.transform(sample)

prediction = model.predict(sample)

if prediction >= 0.5:
    print("Diabetic")
else:
    print("Not Diabetic")
▶️ How to Run the Project
Step 1: Clone Repository
git clone your-repository-link
Step 2: Install Required Libraries
pip install pandas numpy matplotlib scikit-learn tensorflow
Step 3: Run Jupyter Notebook
jupyter notebook
Step 4: Open Notebook
ann_diabetes_project_task.ipynb
📁 Project Structure
├── diabetes.csv
├── ann_diabetes_project_task.ipynb
├── README.md
🎯 Future Improvements
Add Dropout Layers
Hyperparameter Tuning
Improve Model Accuracy
Deploy Using Streamlit
Add Visualization Dashboard
Save and Load Trained Model

Conclusion

This project provides a beginner-friendly implementation of an Artificial Neural Network for diabetes prediction using TensorFlow and Keras. It demonstrates how deep learning can be applied to healthcare-related prediction problems using structured medical data.

The project also helps in understanding:

Data preprocessing
ANN architecture
Binary classification
Model training and evaluation
Prediction using Deep Learning
