import data_module as dm

data = dm.data_upload('./data/', 'car_prices.csv')
print(data.head())

dm.rename_column(data, "Unnamed: 0", "Index")
print(data.head())

print(data.dtypes)
dm.change_type(data, "Index", "float64")
print(data.dtypes)

# Kilometers to Miles
data["Miles_Driven"] = data["Kilometers_Driven"] * 0.621371
print(data.head())

# Data Normalization
print(data[["Miles_Driven", "Kilometers_Driven"]])

dm.data_normalization(data, "Miles_Driven", "Normalized_Miles")    
dm.data_normalization(data, "Kilometers_Driven", "Normalized_Kilometers") 

print(data[["Normalized_Miles", "Normalized_Kilometers"]])