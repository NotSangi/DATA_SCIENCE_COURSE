import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.array([10,12,12,13,12,11,14,13,15,64,12,14,17,19,78,13,59])

plt.boxplot(data, vert=False)
plt.title('Outliers Data Boxplot')
plt.xlabel('Values')
plt.show()

Q1 = np.percentile(data, 25)
Q3= np.percentile(data, 75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR 

filter_data = data[(data >= lower_limit) & (data <= upper_limit)]

print(filter_data)
print(lower_limit)
print(upper_limit)

plt.boxplot(filter_data, vert=False)
plt.title('Data Boxplot Without Outliers')
plt.xlabel('Values')
plt.show()