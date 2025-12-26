# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: AVANESH R
RegisterNumber:25018356

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load dataset
data = pd.read_csv(r"C:\Users\israv\Downloads\Placement_Data.csv")

# Step 2: Preprocessing
# Drop irrelevant columns
if "sl_no" in data.columns:
    data = data.drop("sl_no", axis=1)
if "salary" in data.columns:
    data = data.drop("salary", axis=1)

# Encode categorical columns
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

# Features and target
X = data.drop("status", axis=1).values
y = data["status"].values.reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add bias term (column of 1s)
X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    return -(1/m) * np.sum(y*np.log(h+1e-9) + (1-y)*np.log(1-h+1e-9))

# Gradient Descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        h = sigmoid(X.dot(theta))
        gradient = (1/m) * X.T.dot(h - y)
        theta -= learning_rate * gradient
    return theta

# Initialize parameters
theta = np.zeros((X_train.shape[1], 1))
learning_rate = 0.01
iterations = 1000

# Train model
theta = gradient_descent(X_train, y_train, theta, learning_rate, iterations)

# Predictions
y_pred_prob = sigmoid(X_test.dot(theta))
y_pred = (y_pred_prob >= 0.5).astype(int)

# Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, 
                     index=["Actual:Not Placed", "Actual:Placed"], 
                     columns=["Pred:Not Placed", "Pred:Placed"])
print("\nConfusion Matrix:\n", cm_df)

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("\nClassification Report:\n", report_df.round(2))
*/
```

## Output:

<img width="505" height="301" alt="491673767-57309e7d-e6fa-44d8-9ff7-50f485a5e592" src="https://github.com/user-attachments/assets/2494e459-1e34-4935-b780-ddf607be7f2d" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

