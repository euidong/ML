import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(facecolor='skyblue')

ax = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8])

x = range(1, 6)
y = [[1,4,6,8,9],[2,2,7,10,12],[2,8,5,10,6]]

ax.stackplot(x, y, labels=['A', 'B', 'C'], colors=['orange', 'skyblue', 'grey'])

plt.show()