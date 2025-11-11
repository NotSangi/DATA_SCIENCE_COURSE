import pandas as pd
import numpy as np
import data_module as dm
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = dm.data_upload('./data/', 'car_prices_modified.csv', ',')

print(data.head())

data = pd.get_dummies(data, columns=['Fuel_Type'])

X = data[['Engine', 'Year', 'Fuel_Type_CNG', 'Fuel_Type_Diesel', 'Fuel_Type_LPG', 'Fuel_Type_Petrol']]
y = data[['Power']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

# sns.regplot(x='Engine', y='Power', data=data)
# sns.residplot(x=data['Engine'], y=data['Power'])
# plt.show()

plt.scatter(X_test['Engine'], y_test, label='Data', alpha=0.6)
plt.scatter(X_test['Engine'], prediction, c='red', label='Prediction', alpha=0.6)
plt.xlabel('Engine')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()