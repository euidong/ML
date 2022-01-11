import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(facecolor='skyblue')
ax = fig.add_axes(rect=[0.1,0.1, 0.8, 0.8])

w = 4
h = 3
d = 70


x = np.arange(-2, 2, 0.05)
y = np.arange(-2, 2 ,0.05)

x,y = np.meshgrid(x, y)

z = np.sqrt(x **2 + y ** 2)

cp = ax.contourf(x, y, z)

fig.colorbar(cp)

plt.show()