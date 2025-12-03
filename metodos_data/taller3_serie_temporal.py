import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Serie temporal: 50 mediciones de temperatura
temperaturas = np.random.normal(loc=25, scale=5, size=50)

# Crear Serie con índice simple
serie = pd.Series(temperaturas)

# Descomposición estacional (período de 7 para simular ciclo semanal)
print("\nDescomposición de la serie temporal:")
resultado = seasonal_decompose(serie, model='additive', period=7)

# Graficar descomposición (tendencia, estacionalidad, residuos) 
resultado.plot()
plt.tight_layout()
plt.show()



