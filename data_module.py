import pandas as pd
import numpy as np

def data_upload(route, file):
    data = pd.read_csv(route+file)
    return data

def show_columns(data):
    return data.columns

def save_data(data, route, file):
    data.to_csv(route+file)
    
#Rename all columns
def change_columns(data, columns):
    data.columns = columns
    return data

#Rename a specific column
def rename_column(data, column, new_column):
    data.rename(columns={column:new_column}, inplace=True)

#Change column type
def change_type(data, column, type):
    data[column] = data[column].astype(type)

# Min-Max Scaling
def data_normalization(data, column, new_column):
    data[new_column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())

def stadistics(data, type="Numeric"):
    """
    Type Numeric or All
    """
    
    if type == "All":
        return data.describe(include="all")
    else:
        return data.describe()
    
def replace_nulls(data, column):
    column_mean = data[column].mean()
    data.replace({column: np.nan}, {column:column_mean}, inplace=True)
    return data

if __name__ == '__main__':
    route = "./data/"
    file = "car_prices.csv"
    data = data_upload(route, file)    
    
    print(data.head(5))
    print(show_columns(data))
    
    # save_data(data, route, "copy.csv")
    
    columns_titles = ['Index','Name', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats', 'New_Price']
    change_columns(data, columns_titles)
    print(show_columns(data))
    
    print(stadistics(data))
    
    print(data.info())
    replace_nulls(data, "Seats")
    print(data.info())
    
