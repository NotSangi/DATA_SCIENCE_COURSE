import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = sns.load_dataset('diamonds')

# print(data.head())
# print(data.info())
# print(data.describe())

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.histplot(data['carat'], kde=True)
plt.title('Weight Distribution in Carats')
plt.xlabel('Weight in Carats')
plt.ylabel('Frequency')

plt.subplot(1,2,2)
sns.histplot(data['price'], kde=True)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

#Relation Weight-Price
sns.scatterplot(x='carat', y='price', data=data, alpha=0.6)
plt.title('Relation Weight-Price')
plt.xlabel('Weight (Carats)')
plt.ylabel('Price (dolars)')
plt.show()

values = data['clarity']
sns.boxplot(x='clarity', y='price', data=data, order=values, color='')
plt.title('Price for Clarity')
plt.xlabel('Clarity')
plt.ylabel('Price (dolars)')
plt.show()