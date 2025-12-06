import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report, accuracy_score

data = load_iris()

X = data.data
y = data.target

features = data.feature_names
labels = data.target_names

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.35, random_state=2)

#Creeation and training model
model = DecisionTreeClassifier(random_state=2)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#Evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report: \n {classification_report(y_test, y_pred, target_names=labels)}')

#Visualization
plt.figure(figsize=(12,6))
plot_tree(model, feature_names=features, class_names=labels, filled=True)
plt.title('Decision Tree - Iris Dataset')
plt.show()