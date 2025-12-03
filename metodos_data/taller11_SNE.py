from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X)
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y, cmap='Spectral', s=50)
plt.colorbar()
plt.title('t-SNE de Iris')
plt.show()
