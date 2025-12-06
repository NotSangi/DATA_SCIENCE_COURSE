import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = load_iris()
X = pd.DataFrame(data.data ,columns=data.feature_names)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=2, random_state=0)
model.fit(X_scaled)

#Obtain cluster labels and assign it to the data 
labels = model.labels_
X['cluster'] = labels
print(X.head(5))

#Visualization
plt.scatter(X['sepal length (cm)'], X['sepal width (cm)'], c=X['cluster'], cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Clusters Classification with Kmeans')
plt.show()


