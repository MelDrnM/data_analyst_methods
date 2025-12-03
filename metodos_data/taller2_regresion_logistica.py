import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Crear dataset - Combinar las features en una matriz X
feature1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
feature2 = np.array([11, 21, 31, 41, 51, 61, 71, 81])
X = np.column_stack([feature1, feature2])  # Combinar en matriz

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Dividir datos correctamente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Entrenar modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"Predicciones: {y_pred}")
print(f"Probabilidades: {y_proba}")
print(f"Accuracy: {model.score(X_test, y_test)}")

# Visualizar
plt.scatter(X[:, 0], y, color='blue', label='Datos')
plt.xlabel('Feature 1')
plt.ylabel('Clase')
plt.legend()
plt.show()