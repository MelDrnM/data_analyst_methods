from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

# Cargar dataset de dígitos
digits = load_digits()
X = digits.data  # 1797 muestras, 64 features (imágenes 8x8)
y = digits.target  # 10 clases (0-9)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el árbol de decisión
clf = DecisionTreeClassifier(max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Evaluar el modelo
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print(f"\nAccuracy en entrenamiento: {train_score:.2%}")
print(f"Accuracy en prueba: {test_score:.2%}")

# Predecir primeros 10 ejemplos del conjunto de prueba
print(f"\nPredicciones en primeros 10 ejemplos de prueba:")
y_pred = clf.predict(X_test[:10])
print(f"Predicho: {y_pred}")
print(f"Real:     {y_test[:10]}")

# Visualizar 10 dígitos con sus predicciones
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {y_pred[i]}, Real: {y_test[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()

