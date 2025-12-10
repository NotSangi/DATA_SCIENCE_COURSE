import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

data = {
    'DATE': pd.date_range(start='2025-01-01', periods=10),
    'PRODUCT': ['A','B','C','D','C','A','B','C','A','B'],
    'SALES': [100, 46, 87, 86, 98, 79, 110, 211, 312, 240]    
}

df = pd.DataFrame(data)
df.to_csv('data/sales.csv', index=False)

#Load data from CSV
data = pd.read_csv('data/sales.csv')
print(data.head(10))

#General Dataframe Info
print(data.info())

#Statistics
print(data.describe())

#Count Product Values
print(data['PRODUCT'].value_counts())

#Visualization

    #Bar Graph
plt.bar(df['PRODUCT'], df['SALES'])
plt.title('Sales per product')
plt.xlabel('Product')
plt.ylabel('Sales')
plt.show()

    #Lines Graph - Sales Trend Over Time
plt.plot(df['DATE'], df['SALES'], marker='o')
plt.title('Sales Trends over time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=35)
plt.show()