import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_module as dm

data = dm.data_upload("./data/", "car_prices.csv")
columns_titles = ['Index','Name', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats', 'New_Price']
dm.change_columns(data, columns_titles)

print(data.head())

interval = np.linspace(min(data["Kilometers_Driven"]), max(data['Kilometers_Driven']), 4)
groups_names = ['few', 'normal', 'many']
data['Grouped_Kilometers'] = pd.cut(data['Kilometers_Driven'], interval,  labels=groups_names, include_lowest=True)

print(data['Grouped_Kilometers'])

plt.hist(data['Kilometers_Driven'], bins=interval, rwidth=0.8, color="red")
plt.title('Driven Kilometers Histogram')
plt.xlabel('Kilometers')
plt.ylabel('Frequency')
plt.show()
