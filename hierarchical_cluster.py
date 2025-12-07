import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

X, _ = make_blobs(n_samples=500, centers=5, cluster_std=0.60, random_state=0)
plt.figure(figsize=(10,5))
plt.scatter(X[:,0], X[:,1])   #Scatter that shows the points without labels
plt.show()
print(X)

#Dendrogram Calculation
Z = linkage(X, 'ward')
print(Z)
plt.figure(figsize=(10,5))
plt.title('Dendogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
dendrogram(Z, leaf_rotation=90, leaf_font_size=8)
plt.show()

#Model Training
model = AgglomerativeClustering(n_clusters=5)
labels = model.fit_predict(X)

# plt.figure(figsize=(10,5))
# plt.scatter(X[:,0], X[:,1], c=labels) #Scatter that shows the points with labels
# plt.show()

#Silhouette Index Calculation
silhouette_avg = silhouette_score(X, labels)
print(f'Average Silhouette Index: {silhouette_avg:.2f}')

#Silhouette Calculation for Each Sample
sample_silhouette_values = silhouette_samples(X, labels)

#Silhouette Visualization
plt.figure(figsize=(10,5))
y_lower = 10
for i in range(5):
    ith_cluster_silhouette_values = sample_silhouette_values[labels==i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper=y_lower + size_cluster_i
    color = plt.cm.nipy_spectral(float(i)/5)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10
plt.title('Silhouette Graphic')
plt.xlabel('Silhouette Coeficient')
plt.ylabel('Clusters')    
plt.axvline(x=silhouette_avg, color="red", linestyle="--")
plt.yticks([])
plt.show()