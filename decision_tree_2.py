import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = pd.read_csv('./data/processed_data.csv', header=None)
# print(data.head(10))

# print('Nulls values per Column')
# print(data.isnull().sum())

X = data.iloc[:,:-1] #Select every column except the last one
y = data.iloc[:,-1] #Select only the last column

X.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
print(X.head())

# print(X['ca']) #We can see a ? in the last row
# print(X['thal']) # In this one to

X['ca'] = X['ca'].replace('?', 0.0) #Replace the ? whit 0.0
X['thal'] = X['thal'].replace('?', 0.0) #Also in this one

# print(X.dtypes)
X['ca'] = X['ca'].astype('float64')
X['thal'] = X['thal'].astype('float64')
# print(X.dtypes)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = DecisionTreeClassifier()
# model.fit(x_train, y_train)

# joblib.dump(model, 'models/decision_tree.pkl')

load_model = joblib.load('models/decision_tree.pkl')

y_pred = load_model.predict(x_test)

x_test_whit_predictions = x_test
x_test_whit_predictions['prediction'] = y_pred
x_test_whit_predictions['true_value'] = y_test

print('predictions vs true target value')
print(x_test_whit_predictions.head(10))

print(f'Accuracy Score: {round(accuracy_score(y_test, y_pred), 2)}') #Evaluate the global precisión of the model (1 -> all corrects, 0 -> all wrong)
print('Classification Report')
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8,6))
plt.scatter(x_test_whit_predictions['true_value'], x_test_whit_predictions['prediction'], color='blue', alpha=0.4)
plt.xlabel('True Value')
plt.ylabel('Prediction Value')
plt.title('True Values v Prediction Values')
plt.show()
