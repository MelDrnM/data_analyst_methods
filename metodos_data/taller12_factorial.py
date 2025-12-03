import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer

np.random.seed(42)

n_samples = 200

# Factores latentes
factor_mat = np.random.randn(n_samples, 2)
factor_verbal = factor_mat[:, 0]
factor_numerico = factor_mat[:, 1]

# Variables observadas (con algo de ruido)
df = pd.DataFrame({
    'Algebra': 0.8 * factor_numerico + np.random.randn(n_samples) * 0.3,
    'Geometria': 0.85 * factor_numerico + np.random.randn(n_samples) * 0.25,
    'Estadistica': 0.75 * factor_numerico + np.random.randn(n_samples) * 0.35,
    'Lectura': 0.8 * factor_verbal + np.random.randn(n_samples) * 0.3,
    'Escritura': 0.82 * factor_verbal + np.random.randn(n_samples) * 0.28,
    'Vocabulario': 0.78 * factor_verbal + np.random.randn(n_samples) * 0.32
})

print("Primeras filas del dataset:")
print(df.head())

# Calcular matriz de correlaciones
print("\n" + "="*60)
print("MATRIZ DE CORRELACIONES")
print("="*60)
corr_matrix = df.corr()
print(corr_matrix.round(3))

# Visualizar la matriz de correlaciones
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlaciones - Habilidades Académicas')
plt.tight_layout()
plt.show()

# Aplicar análisis factorial
print("\n" + "="*60)
print("ANÁLISIS FACTORIAL")
print("="*60)

fa = FactorAnalyzer(n_factors=2, rotation='varimax')
fa.fit(df)

# Cargas factoriales
loadings_df = pd.DataFrame(
    fa.loadings_,
    index=df.columns,
    columns=["Factor 1", "Factor 2"]
)
print("\nCargas Factoriales (después de rotación Varimax):")
print(loadings_df.round(3))

# Varianza explicada por cada factor
variance = fa.get_factor_variance()
print("\nVarianza explicada por cada factor:")
print(f"  Varianza: {variance[0].round(3)}")
print(f"  Varianza proporcional: {variance[1].round(3)}")
print(f"  Varianza acumulada: {variance[2].round(3)}")

# Comunalidades
print("\nComunalidades (proporción de varianza explicada para cada variable):")
communalities = pd.DataFrame({
    'Variable': df.columns,
    'Comunalidad': fa.get_communalities()
})
print(communalities.round(3))

# Visualizar las cargas factoriales
plt.figure(figsize=(10, 6))
loadings_df.plot(kind='bar', width=0.8)
plt.title('Cargas Factoriales por Variable')
plt.xlabel('Variables')
plt.ylabel('Carga Factorial')
plt.xticks(rotation=45)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.legend(title='Factores')
plt.tight_layout()
plt.grid(axis='y', alpha=0.3)
plt.show()

