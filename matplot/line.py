import matplotlib.pyplot as plt
import numpy as np
import math

values = [1,5,8,9,7,13,6,1,8]

fig = plt.figure(facecolor="skyblue")

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], xlabel='keys', ylabel='values')

x = np.arange(0, math.pi * 2.5, 0.05)

y = np.sin(x) * 5 + 5

ax.grid()

ax.plot(values, color='red', marker='o')

plt.show()
