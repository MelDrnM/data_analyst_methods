import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

# Cargar California Housing dataset
print("Descargando California Housing dataset...")
data = fetch_california_housing()
X = data.data  # 8 features
y = data.target  # Precios de viviendas

print(f"Dataset shape: {X.shape}")
print(f"Features: {data.feature_names}")
print(f"Target (precios): min=${y.min():.2f}, max=${y.max():.2f}")

# Escalar datos (importante para PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA para reducir de 8 features a 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nComponentes principales (primeras 5 muestras):")
print(X_pca[:5])
print(f"\nVarianza explicada por cada componente: {pca.explained_variance_ratio_}")
print(f"Varianza total explicada: {pca.explained_variance_ratio_.sum():.2%}")

# Visualizar en 2D
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6, s=10)
plt.colorbar(scatter, label='Precio de vivienda')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)")
plt.title("Visualización PCA del dataset California Housing")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

