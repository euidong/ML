#importing libraries
import numpy as np
import matplotlib.pyplot as plt

## grouped bar 
data = [[30, 25, 50, 20],
        [40, 23, 51, 17],
        [35, 22, 45, 19]]

label=['Match-1','Match-2','Match-3','Match-4']

X = np.arange(len(label))

y=[0.25, 1.25, 2.25, 3.25]

fig = plt.figure(facecolor='skyblue')
ax = fig.add_axes([0.15,0.1,0.8,0.8])

ax.set_xlabel('Match')
ax.set_ylabel('Score')
ax.set_title('Comparing scores of three players in different cricket matches')
ax.set_xticks(ticks=y)
ax.set_xticklabels(label)

#creating bar plots
ax.bar(X + 0.00, data[0], color = 'black', width = 0.25,alpha=0.5,label='Ricky Ponting')
ax.bar(X + 0.25, data[1], color = 'orange', width = 0.25,alpha=0.6,label='Sachin Tendulkar')
ax.bar(X + 0.50, data[2], color = 'red', width = 0.25,alpha=0.5,label='Adam Gilchrist')

ax.legend()

## stacked bar

import numpy as np
import matplotlib.pyplot as plt

male = [20, 35, 30, 35, 27]
female = [25, 32, 34, 20, 25]

ind = np.arange(5)    # the x locations for the groups

fig2 = plt.figure(facecolor='skyblue')
ax2 = fig2.add_axes(rect=[0.15, 0.1, 0.8, 0.8])

ax2.bar(ind, male, width= 0.35 ,color='black',alpha=0.8,label='Male')
ax2.bar(ind, female, width= 0.35 ,bottom=male,color='orange',label='Feamle')

ax2.set_ylabel('Age(in Years)')
ax2.set_title('Age of employees by group and gender')

ax2.set_xticks(ind)
ax2.set_xticklabels(['G1', 'G2', 'G3', 'G4', 'G5'])
ax2.set_yticks(np.arange(0, 81, 10))

ax2.legend()
plt.show()