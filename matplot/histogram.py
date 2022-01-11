import matplotlib.pyplot as plt
import numpy as np

scores = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
fig = plt.figure(facecolor='skyblue')
ax = fig.add_axes(rect=[0.15, 0.1, 0.8, 0.8], xlabel='Score', ylabel='No. of students', title="Student's score")

ax.hist(scores, bins=[0, 20, 40, 60, 80, 100], color='skyblue', edgecolor='black', alpha=0.4)

plt.show()