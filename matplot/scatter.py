import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(facecolor='skyblue')
ax = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8])

np.random.seed(0)

n = 50
x = np.random.rand(n)
y = np.random.rand(n)

plt.scatter(x, y)

plt.show()
