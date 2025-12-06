import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Data
X, _ = make_moons(n_samples=1000, noise=0.05)

model = DBSCAN(eps=0.2, min_samples=5)
model.fit(X)

labels = model.labels_
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()