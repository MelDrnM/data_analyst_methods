import numpy as np
import matplotlib.pyplot as plt

x = np.array([2, 4, 6, 8, 10])
y = np.array([3, 7, 5, 11, 14])
             
x_mean = np.mean(x)
y_mean = np.mean(y)
β1 = np.sum((x - x_mean)*(y - y_mean)) / np.sum((x - x_mean)**2)
β0 = y_mean - β1 * x_mean
print(f'β0 = {β0}')
print(f'β1 = {β1}')

plt.scatter(x, y, color='blue', label='Datos')
plt.plot(x, β0 + β1 * x, color='red', label='Modelo')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
