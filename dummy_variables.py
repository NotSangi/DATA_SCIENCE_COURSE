import pandas as pd
import data_module as dm

data = dm.data_upload('./data/', 'car_prices.csv')
dm.rename_column(data, "Unnamed: 0", "Index")
print(data.columns)

# Transform a categorical variable into dummys
dummy_column = pd.get_dummies(data['Fuel_Type'])
print(dummy_column)

data = pd.get_dummies(data, columns=['Transmission', 'Fuel_Type'])
print(data.head())
print("-"*45)



