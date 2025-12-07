import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

pca = PCA(n_components=2)
x_pca = pca.fit_transform(X)

print(X)
print('-'*25)
print(x_pca)

#Visualization

plt.scatter(x_pca[:,0], x_pca[:,1], c=y, cmap='viridis')
plt.title('PCA')
plt.xlabel('Main Component 1')
plt.ylabel('Main Component 2')
plt.show()