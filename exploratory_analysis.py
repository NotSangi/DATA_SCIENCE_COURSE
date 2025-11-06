import data_module as dm
import matplotlib.pyplot as plt
from scipy import stats

data = dm.data_upload('./data/', 'car_prices_2.csv', ';')
# print(data.head())

dm.rename_column(data, "Unnamed: 0", "Index")
# print(data.head())

data.set_index("Index", inplace=True)
# print(data.head())

# print(dm.show_columns(data))

print(dm.stadistics(data, "All"))

# Null Values
data = dm.replace_nulls(data, "Seats")
print(data)

# Change Types
print(data['Engine'])
print(data['Power'])

data["Engine"] = (data["Engine"].str.extract(r"(\d+\.?\d*)").astype(float)) # This line retrieves only the integer and decimal values ​​from the column.
data["Power"] = (data["Power"].str.extract(r"(\d+\.?\d*)").astype(float)) 

print(data['Engine'])
print(data['Power'].describe())

# Data Visualization

plt.boxplot([data['Power'].dropna()])
plt.title("Power")
plt.show()

# Scatter plot with var relations

x = data['Engine']
y = data['Power']

plt.scatter(x, y)
plt.title("Power - Engine")
plt.xlabel("Engine")
plt.ylabel("Power")
plt.show()

# Ratio Coefficient and P Value

data = dm.replace_nulls(data,"Engine")
data = dm.replace_nulls(data,"Power")

pearson_coef, p_value = stats.pearsonr(data["Engine"], data["Power"])
print(f'The Pearson Coefficiente is: {pearson_coef}')
print(f'The P Value is: {p_value:.2f}')

