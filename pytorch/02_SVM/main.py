from functools import reduce
from numpy import float64
import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# * 1. import data
df = pd.read_csv('data/two_centroid_data.csv')
data = df.to_numpy(dtype=float64)
data = torch.from_numpy(data).float()
X = data[:, :2]
Y = data[:, 2]

# * 2. preprocessing
X = (X - X.mean()) / X.std()
Y[Y == 0] = -1


class Dataset(torch.utils.data.Dataset):
    """Some Information about Dataset"""

    def __init__(self, x, y):
        super(Dataset, self).__init__()
        self.x = x
        self.y = y
        self.len = len(x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


class SVM(nn.Module):
    """Some Information about SVM"""

    def __init__(self, input_size, output_size):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        yhat = self.linear(x)
        return yhat


learning_rate = 0.1
epochs = 10
batch_size = 1

dataset = Dataset(X, Y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

model = SVM(2, 1)
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def criterion(y, yhat): return torch.mean(torch.clamp(1 - yhat * y, min=0))


LOSS = []
for epcoh in range(epochs):
    epoch_loss = 0
    for x, y in dataloader:
        yhat = model(x)
        optimizer.zero_grad()
        loss = criterion(y, yhat)
        loss.backward()
        optimizer.step()
        epoch_loss += loss
    LOSS.append(epoch_loss.data.item())

weight = model.linear.weight.detach().numpy()[0]
bias = model.linear.bias.detach().numpy()


def line(x, offset):
    return (offset - 1 * (weight[0] * x + bias)) / weight[1]


x = np.arange(-2, 2, 1)
yu = line(x, 1)
y = line(x, 0)
yd = line(x, -1)

fig = plt.figure(facecolor='skyblue')
ax1 = fig.add_subplot(1, 2, 1, title='data')
ax2 = fig.add_subplot(1, 2, 2, title='cost')

ax1.scatter(X[:, 0], X[:, 1])
ax1.plot(x, y)
ax1.plot(x, yu)
ax1.plot(x, yd)
ax1.fill_between(x, y1=yu, y2=y, color='lightpink', alpha=0.5)
ax1.fill_between(x, y1=y, y2=yd, color='skyblue', alpha=0.5)

ax2.plot(np.arange(0, epochs), LOSS)

plt.show()
