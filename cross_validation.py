import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

tips = sns.load_dataset('tips')

print(tips)
# print('Basic Stadistics \n', tips.describe())
# print('Data Types \n',tips.dtypes)
# print('Basic Info \n', tips.info())

#Data Cleaning
clean_tips = tips.dropna().drop_duplicates()

print('Basic Info Cleaned Data')
print(clean_tips.info())

sns.heatmap(clean_tips.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Vars Correlation')
plt.show()

X = clean_tips.drop('tip', axis=1)
X = X.drop('time', axis=1)
y = clean_tips['tip']

model = Pipeline([
    ('preprocessor', ColumnTransformer(
        [
            ('scaler', StandardScaler(),['total_bill', 'size']),
            ('onehot', OneHotEncoder(), ['day', 'sex', 'smoker'])
        ], 
        remainder='passthrough'
    )),
    ('regressor', RandomForestRegressor())
])

scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
rmse_scores = np.sqrt(-scores)
print('\n Cross Validation Results:')
print('RMSE Mean: ', rmse_scores.mean())
print('Standar Deviation: ', rmse_scores.std())
model.fit(X, y)

# joblib.dump(model, 'models/random_forest_model.pkl')
# print('Model Saved in File: ', 'models/random_forest_model.pkl')

loaded_model = joblib.load('models/random_forest_model.pkl')

x_new = X.head(5)
print(x_new)
predictions = loaded_model.predict(x_new)
print('Prediction for First 5: ')
print(predictions)
