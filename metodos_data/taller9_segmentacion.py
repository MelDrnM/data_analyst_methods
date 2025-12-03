import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random

def generar_cliente(id_cliente):
    return {
        "edad": random.randint(18,70),
        "valor": random.randint(50000, 200000),
        "cantidad": random.randint(1, 10)
    }

clientes = [generar_cliente(i) for i in range(1, 21)]

df = pd.DataFrame(clientes)

print(df)

kmeans = KMeans(n_clusters=3, random_state=0)
df['Segmento'] = kmeans.fit_predict(df)
print(df)

plt.scatter(df['edad'], df['valor'], c=df['Segmento'], cmap='viridis')
plt.xlabel('edad')
plt.ylabel('valor')
plt.title('Segmentación de Clientes')
plt.show()

