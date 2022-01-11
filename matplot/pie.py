import matplotlib.pyplot as plt
import numpy as np

labels = ['English', 'Spanish', 'Russian', 'French', 'Chinese']
sizes = [55, 15, 8, 10, 12]
colors = {'grey', 'orange', 'red', 'white', 'pink'}

explode = [0, 0, 0, 0.2, 0]

fig = plt.figure(facecolor='skyblue')
ax = fig.add_axes(rect=[0.1,0.1,0.8,0.8], title='Language Distribution')


# @oarams
# autpct: pie안에 담기는 내용
ax.pie(
  sizes, 
  explode=explode, 
  colors=colors, 
  labels=labels, 
  autopct='%.2f%%', 
  shadow=True, 
  startangle=90, 
  radius=1)

ax.axis('equal')

plt.show()
