import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

#Read Train and Test CSV
train_data = pd.read_csv('data/titanic/train.csv')
test_data = pd.read_csv('data/titanic/test.csv')

# print(train_data.head(5))
# print(train_data.info())
# print(train_data.describe())

#Get the data to train the model
X = train_data.drop(['PassengerId','Name', 'Ticket', 'Survived', 'Cabin', 'Embarked'], axis=1)
y = train_data['Survived']

#Fill both datasets NA with the Mean value
fare_mean = train_data['Fare'].mean()
train_data['Fare'].fillna(fare_mean, inplace=True)
test_data['Fare'].fillna(fare_mean, inplace=True)

#Model Creation
model = Pipeline([
    ('preprocessor', ColumnTransformer(
        [
            ('onehot', OneHotEncoder(), ['Sex'])
        ], remainder='passthrough'
    )),
    ('decision_tree', RandomForestClassifier(n_estimators=100, random_state=0, max_depth=10))
])

#Model Training
model.fit(X, y)

#Get the data to test the model
X_test = test_data.drop(['PassengerId','Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

#Model Prediction
prediction = model.predict(X_test)

#Get the true Survivers CSV
true_data = pd.read_csv('data/titanic/gender_submission.csv') 

#Test Dataset Copy and insert the Real and Predicted Values
df = test_data.copy()
df['Prediction'] = prediction
real_values = true_data['Survived']
df['Real'] = real_values

#Model Evaluation
cr = classification_report(real_values, prediction)
print('-'*25)
print(df.head(15))
print('-'*25)
print('Classification Report \n', cr)
print('-'*25)

#In Case we Want to save the Copy in a CSV

# df.to_csv('data/titanic/prediction.csv')