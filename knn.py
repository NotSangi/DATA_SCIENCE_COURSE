import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_breast_cancer
import joblib

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

# KNN MODEL
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train_scaled, y_train)
knn_prediction = knn_model.predict(x_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_prediction)
print('Accuracy KNN Neighbors: ', knn_accuracy)
print('Classification Report Knn \n', classification_report(y_test, knn_prediction))

# RANDOM FOREST MODEL
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(x_train_scaled, y_train)
rf_prediction = rf_model.predict(x_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_prediction)
print('Accuracy Random Forest: ', rf_accuracy)
print('Classification Report RF \n', classification_report(y_test, rf_prediction))

#BEST MODEL 
best_model = knn_model if knn_accuracy > rf_accuracy else rf_model
# joblib.dump(best_model, 'models/best_model.pkl')

#MODEL LOAD
load_model = joblib.load('models/best_model.pkl')
print('Best Model is: ', load_model)
pred=load_model.predict(x_test_scaled)
compare=pd.DataFrame({'Real': y_test, 'Prediction': pred})

print('Real Values v Predicted Values')
print(compare)