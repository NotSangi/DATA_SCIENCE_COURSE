from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

data = load_iris()
X, y = data.data, data.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Pipeline creation
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier())
])

#Pipeline Training
pipeline.fit(x_train, y_train)
evaluation = pipeline.score(x_test, y_test)
print(f'Accuracy: {evaluation}')

#Model Saving
# joblib.dump(pipeline, 'models/pipeline.pkl')

#Model Loading
loaded_model = joblib.load('models/pipeline.pkl')

example = x_test[0].reshape(1,-1)
print('Example data', example)
prediction = loaded_model.predict(example)
print('Prediction Value Index: ', prediction)
print(f'Prediction: {data.target_names[prediction]}')