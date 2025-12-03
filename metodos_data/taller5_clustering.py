import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# Crear datos de ejemplo
np.random.seed(42)
X = np.vstack((
    np.random.normal(loc=0.0, scale=1.0, size=(50, 2)),
    np.random.normal(loc=5.0, scale=1.0, size=(50, 2)),
    np.random.normal(loc=2.5, scale=1.0, size=(50, 2))
))

# Crear el modelo KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
# Ajustar el modelo
kmeans.fit(X)
# Obtener etiquetas y centroides
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Graficar los clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.title('Clusters y sus centroides')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.show()