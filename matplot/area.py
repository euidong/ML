import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(facecolor='skyblue')

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

area_values_1 = [12, 17, 14, 15, 20, 28, 30, 28, 10]
area_values_2 = [15, 13, 14, 25, 30, 26, 22, 18, 18]

ax.fill_between(np.arange(9), area_values_1, color='skyblue', alpha=0.5)
ax.plot(np.arange(9), area_values_1, color='black', alpha=0.6, linewidth=2)

ax.fill_between(np.arange(9), area_values_2, color='lightpink', alpha=0.5)
ax.plot(np.arange(9), area_values_2, color='black', alpha=0.6, linewidth=2)

plt.show()