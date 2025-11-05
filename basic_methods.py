import pandas as pd
import numpy as np

file = "./data/car_prices.csv"
data = pd.read_csv(file)

header_titles = ['Index','Name', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats', 'New_Price']
data.columns = header_titles

print("-"*45)
print(data.dtypes) # Displays the type of values ​​in the columns
print("-"*45)
print(data.describe()) # Displays information from numeric columns
print("-"*45)
print(data.describe(include="all")) # Displays information from all columns
print("-"*45)
print(data.info()) # Displays info such as the count of non-null values

# Data preprocessing
print("-"*45)
print(data["Seats"].head(5))
data["Seats"] = data["Seats"] + 1
print(data["Seats"].head(5))

print("-"*45)
print(data.describe())

# Null Values Substitution
# print("-"*45)
# data.dropna(subset=["Seats"], axis=0, inplace=True)
# print(data.describe())

# print("-"*45)
# data.replace(np.nan, "nulo")
# print(data.describe())
# print(data["Seats"] > 5)

print("-"*45)
seats_mean = data["Seats"].mean()
print(seats_mean)
data.replace({"Seats": np.nan}, {"Seats":seats_mean}, inplace=True)
print(data.describe())
