import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = sns.load_dataset('iris')
print(data)
print(data.isnull().sum())
print(data.dtypes)

data = pd.get_dummies(data, columns=['species']) #Convert the categoriacal variable into boolean columns
print(data.head()) 
correlation_matrix = data.corr() #Correlation Matrix (+ -> Positive Correlation - Grows or Drecrease at the same time) (- -> Negative Correlation - one Decrease, the other Grows )
print(correlation_matrix) 

plt.figure(figsize=(7, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm') #Graph a heatmap showing correlations
plt.title('Correlation Matrix')
plt.show()

data = sns.load_dataset('iris')

X = data.drop('species', axis=1)
y = data['species']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# model = RandomForestClassifier(n_estimators=100, random_state=42) # Model Creation with 
# model.fit(x_train, y_train)
# joblib.dump(model, 'models/model_random_forest.pkl')

load_model = joblib.load('models/model_random_forest.pkl') #Import the model

new_data = [[5.1, 3.5, 1.4, 0.2]]
y_pred = load_model.predict(new_data)
print('Prediction: ', y_pred)

y_pred_test = load_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred_test)
print('Accuracy: ', accuracy)
print('Classification Report \n', classification_report(y_test, y_pred_test))