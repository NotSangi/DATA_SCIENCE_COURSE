import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target 

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for k in range(1,20):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f'K = {k}, Accuracy: {model.score(x_test, y_test)}') #Shows the cuantity of neighbors and the prediction accuracy

# print('Classification Report \n', classification_report(y_test, y_pred))
# print('Confusión Matrix \n', confusion_matrix(y_test, y_pred)) #Shows how many were predicted correctly

