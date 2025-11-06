# y = a + bx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Create a dataset with expenses in advertising and sales
data = {
    'Expenses' : [100, 200, 300, 400, 536, 234, 413 ,552, 664, 892, 381, 849, 912, 400, 350, 109],
    'Sales' : [134, 210, 320, 414, 546, 254, 433 ,562, 674, 902, 391, 859, 922, 410, 360, 120]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Separate the predictor and target vars 
X = df[['Expenses']] #If we use only one [], it returns a Series, but the sk methods use matrix/df
y = df[['Sales']]

print(type(X))
print(type(y))

# Separate the train and test vars
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = LinearRegression()
model.fit(X_train, y_train)

#Model Prediction
y_pred = model.predict(X_test)

# Coefficient and Intercept
print(f"Coeficient: {model.coef_[0]}")
print(f"Intersect: {model.intercept_}")

#Metrics 
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.3f}")

results = {
    "Real": y_test.to_numpy().ravel(),
    "Pred": y_pred.ravel()
}
df_results = pd.DataFrame(results)
print(df_results)

# Visualization

plt.scatter(X_test, y_test, color="Blue")
plt.plot(X_test, y_pred, color="red", linewidth=1)
plt.title("Model Prediction in Expenses vs Sales")
plt.xlabel("Expenses")
plt.ylabel("Sales")
plt.show()



