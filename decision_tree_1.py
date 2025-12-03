import data_module as dm
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

data = dm.data_upload('./data/', 'melb_data.csv')
data.drop('Unnamed: 0', axis=1, inplace=True)
# print(data.head(10))
# print(data.columns)

y = data['Price']
selected_columns = ['Rooms','Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = data[selected_columns]

imputer = SimpleImputer()
X = pd.DataFrame(imputer.fit_transform(X)) # It could delete the column names
X.columns = selected_columns
# print(X.describe())

# object = (X.dtypes == 'object')
# print(object)

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)

# model = DecisionTreeRegressor()
# model.fit(x_train, y_train)

# prediction = model.predict(x_test)
# rounded = np.round(prediction, decimals=1)

# x_test['Prediction'] = rounded
# x_test['Real'] = y_test

# print(x_test.head(10))

# print(f'Mean Absolute Error: {mean_absolute_error(y_test, rounded)}') 


def give_mae(max_nodes, x_train, x_test, y_train, y_test):
    model = DecisionTreeRegressor(max_leaf_nodes=max_nodes, random_state=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae

for max_nodes in [10, 50, 100, 150, 200, 1000, 5000]:
    mae = give_mae(max_nodes, x_train, x_test, y_train, y_test)
    print('For {} nodes \t MAE: {}'.format(max_nodes, mae))