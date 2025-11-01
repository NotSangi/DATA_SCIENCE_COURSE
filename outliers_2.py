import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.normal(0, 0.5, 100)
y = np.random.normal(0, 0.5, 100)
x[98:] = [3,-3]
y[98:] = [3,-3]

plt.scatter(x,y)
plt.title("Outliers Scatter plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

center_distance = np.sqrt(x**2 + y**2)
limit = np.percentile(center_distance, 95) # Umbral del 95%

x_filtered = x[center_distance <= limit]
y_filtered = y[center_distance <= limit]

print('-'*35)
print(center_distance)
print(limit)
print(x_filtered)
print(y_filtered)

plt.scatter(x_filtered,y_filtered)
plt.title("Scatter plot without outliers")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

