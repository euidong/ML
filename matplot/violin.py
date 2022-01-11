import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(facecolor='skyblue')
ax = fig.add_axes(rect=[0.1,0.1, 0.8, 0.8])

np.random.seed(1)

values = np.random.normal(100, 10, 200)

ax.violinplot(values, vert=True, showmedians=True)

plt.show()