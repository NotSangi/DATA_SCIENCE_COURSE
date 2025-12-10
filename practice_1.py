import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae

tips = sns.load_dataset('tips')
print(tips.head())

# sns.barplot(x='day', y='total_bill', data=tips, hue='size')
# plt.show()

# sns.boxplot(x='day', y='tip', data=tips, hue='sex')
# plt.show()

# sns.scatterplot(x='total_bill', y='tip', data=tips)
# plt.show()

# sns.histplot(tips['total_bill'])
# plt.show()

# sns.heatmap(tips.corr(numeric_only=True))
# plt.show()

#Categorical Columns

categories = (tips.dtypes == 'category')
print(categories)

dummy = pd.get_dummies(tips)
print(dummy.columns)

dummy.dropna().drop_duplicates()

model = LinearRegression()

X = dummy.drop('tip', axis=1)
y = dummy['tip']

print(X)
print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.30, random_state=42)
model.fit(x_train, y_train)

prediction = model.predict(x_test)

df = pd.DataFrame({
    'True Value': y_test,
    'Prediction': prediction
})

print(df)

MAE = mae(y_test, prediction)
print(MAE)

plt.scatter(tips['total_bill'], tips['tip'], c="blue", label="REAL", alpha=0.7)
plt.scatter(x_test['total_bill'], prediction, c="red", label="PREDICTION", alpha=0.7)
plt.xlabel('Total Bill')
plt.ylabel('Tips')
plt.legend()
plt.show()