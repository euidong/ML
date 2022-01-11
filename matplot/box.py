import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(facecolor='skyblue')
ax = fig.add_axes(rect = [0.1, 0.1, 0.8, 0.8])

values = [np.random.normal(100, 10, 200) for _ in range(3)]

ax.boxplot(values, notch=True, labels=['random1', 'random2', 'random3'])

plt.show()