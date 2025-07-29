import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

# Data
train_df = pd.read_csv('mnist_train.csv')
test_df = pd.read_csv('mnist_test.csv')

X_train = train_df.iloc[:, 1:].values   # pixel values
y_train = train_df.iloc[:, 0].values    # digit labels

X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

X_train = X_train / 255.0
X_test = X_test / 255.0

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)
# EVALUATE 
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, log_preds):.4f}")

sgd_model = SGDClassifier(loss='log_loss', max_iter=1000)
sgd_model.fit(X_train, y_train)
sgd_preds = sgd_model.predict(X_test)
# EVALUATE 
print(f"SGDClassifier Accuracy: {accuracy_score(y_test, sgd_preds):.4f}")

# KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)
# EVALUATE 
print(f"KNN Accuracy: {accuracy_score(y_test, knn_preds):.4f}")


# Logistic Regression Accuracy: 0.9216
# SGDClassifier Accuracy: 0.8973
# KNN Accuracy: 0.9452