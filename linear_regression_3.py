import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import data_module as dm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# data = dm.data_upload('./data/', 'housing.csv')
# print(data.head(10))

# correlation_matrix = data.corr().round(2)
# plt.figure(figsize=(12,8))
# sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm')
# plt.title("Correlation Beetween Characteristics")
# plt.show()

# X = data.drop('MEDV',axis=1)
# y = data['MEDV']

scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# print(X_scaled)

# x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# model = LinearRegression()
# model.fit(x_train, y_train)

# y_pred = model.predict(x_test)
# print(y_pred)

# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print('Mean Squared Error: ', round(mse,2))
# print('R2 Score: ', round(r2,2))


# joblib.dump(model, "models/linear_regression.pkl")

model = joblib.load('models/linear_regression.pkl')

new_data = np.array([[0.002, 18.3, 2.31, 5.63, 0, 6.757, 0.043, 2.65, 3.56, 0.2,4.56, 7.98, 32.1]])

scaled_data = scaler.fit_transform(new_data)

prediction = model.predict(scaled_data)
print(prediction)

