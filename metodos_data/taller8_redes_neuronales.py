import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Datos de ejemplo - Problema XOR
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])  # XOR

print("=" * 60)
print("COMPARACIÓN DE FUNCIONES DE ACTIVACIÓN - Problema XOR")
print("=" * 60)
print("\nEntradas:")
print(X)
print("Salidas esperadas:")
print(y)

# Funciones de activación a probar
funciones_activacion = ['relu', 'tanh', 'sigmoid', 'elu', 'selu']

resultados = []

for activacion in funciones_activacion:
    print("\n" + "=" * 60)
    print(f"PROBANDO CON FUNCIÓN DE ACTIVACIÓN: {activacion.upper()}")
    print("=" * 60)

    # Crear modelo con la función de activación actual
    model = Sequential([
        Dense(8, input_dim=2, activation=activacion),   # Capa oculta 1
        Dense(8, activation=activacion),                 # Capa oculta 2
        Dense(4, activation=activacion),                 # Capa oculta 3
        Dense(1, activation='sigmoid')                   # Capa de salida (siempre sigmoid para clasificación binaria)
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Entrenar el modelo
    print(f"\nEntrenando con {activacion}...")
    history = model.fit(X, y, epochs=1000, verbose=0)

    # Evaluar
    loss, accuracy = model.evaluate(X, y, verbose=0)

    print(f"\nResultados:")
    print(f"  Pérdida: {loss:.4f}")
    print(f"  Precisión: {accuracy*100:.2f}%")

    # Hacer predicciones
    predicciones = model.predict(X, verbose=0)
    print("\n  Predicciones detalladas:")
    correctas = 0
    for i in range(len(X)):
        pred_class = int(predicciones[i] > 0.5)
        correcto = "✓" if pred_class == y[i] else "✗"
        if pred_class == y[i]:
            correctas += 1
        print(f"  {X[i]} -> {predicciones[i][0]:.4f} -> {pred_class} (esperado: {y[i]}) {correcto}")

    # Guardar resultados
    resultados.append({
        'activacion': activacion,
        'loss': loss,
        'accuracy': accuracy,
        'correctas': correctas
    })

# Resumen comparativo
print("\n" + "=" * 60)
print("RESUMEN COMPARATIVO")
print("=" * 60)
print(f"{'Función':<12} {'Pérdida':<12} {'Precisión':<12} {'Aciertos'}")
print("-" * 60)
for r in resultados:
    print(f"{r['activacion']:<12} {r['loss']:<12.4f} {r['accuracy']*100:<12.2f}% {r['correctas']}/4")

# Mejor resultado
mejor = max(resultados, key=lambda x: (x['accuracy'], -x['loss']))
print("\n" + "=" * 60)
print(f"MEJOR FUNCIÓN DE ACTIVACIÓN: {mejor['activacion'].upper()}")
print(f"Precisión: {mejor['accuracy']*100:.2f}%")
print(f"Pérdida: {mejor['loss']:.4f}")
print("=" * 60)
