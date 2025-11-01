import pandas as pd

file = "./data/car_prices.csv"
data = pd.read_csv(file)

header_titles = ['Index','Name', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats', 'New_Price']
data.columns = header_titles
print(data.head(10))

# data['New_Price'] = data['New_Price'].str.replace('Lakh', '', regex=False)
# data['New_Price'] = data['New_Price'].str.replace('Cr', '', regex=False)
# data['New_Price'] = data['New_Price'].astype(float)

# mean = data['New_Price'].mean()
# data.fillna({'New_Price': mean}, inplace=True)
# print(data.head(10))

# file = "./data/car_prices.json"
# data.to_json(file)
# pd.read_json(file)


