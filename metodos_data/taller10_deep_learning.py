import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os


def audio_a_espectrograma(path):
    y, sr = librosa.load(path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Redimensionar a 28x28 como MNIST
    S_dB = librosa.util.fix_length(S_dB, size=28, axis=1)
    S_dB = librosa.util.fix_length(S_dB, size=28, axis=0)

    return S_dB.astype("float32")


# ================================
# 3. RUTAS DE AUDIOS Y ETIQUETAS
# ================================
# Asegúrate que tu carpeta media tenga estos nombres:
frecuencias = [440, 600, 800, 1000, 1200]
rutas = [f"media/beep_{f}.wav" for f in frecuencias]

# Etiquetas → cada frecuencia es una clase diferente
y = np.array([i for i in range(len(frecuencias))])


# ================================
# 4. CREAR EL DATASET (X)
# ================================
X = np.array([audio_a_espectrograma(r) for r in rutas])
X = X[..., np.newaxis]  # (N, 28, 28, 1)

print("Forma de X:", X.shape)
print("Forma de y:", y.shape)


# ================================
# 5. DEFINIR EL MODELO CNN
# ================================
model = models.Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(frecuencias), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# ================================
# 6. ENTRENAR EL MODELO
# ================================
model.fit(X, y, epochs=20, verbose=1)


# ================================
# 7. PROBAR EL MODELO
# ================================
preds = model.predict(X)
print("\nPredicciones:", preds.argmax(axis=1))
print("Etiquetas reales:", y)
