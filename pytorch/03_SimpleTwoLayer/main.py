import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn


class Dataset(torch.utils.data.Dataset):
    """Some Information about Dataset"""

    def __init__(self):
        super(Dataset, self).__init__()
        self.X = torch.arange(-2, 2, 0.1).view(-1, 1)
        self.Y = torch.zeros(self.X.shape[0])
        self.Y[10:30] = 1
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


class SimpleTwoLayer(nn.Module):
    """Some Information about SimpleTwoLayer"""

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleTwoLayer, self).__init__()
        self.first_layer = nn.Linear(input_size, hidden_size)
        self.second_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        z = self.first_layer(x)
        u = torch.sigmoid(z)
        yhat = torch.sigmoid(self.second_layer(u))
        return yhat


def criterion(yhat, y):
    out = -1 * torch.mean(y * torch.log(yhat) +
                          (1 - y) * torch.log(1 - yhat))
    return out


dataset = Dataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

model = SimpleTwoLayer(1, 2, 1)
model.train()

learning_rate = 0.1
epochs = 1000
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

LOSS = []
for _ in range(epochs):
    epoch_loss = 0
    for x, y in dataloader:
        yhat = model(x)
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
    LOSS.append(epoch_loss)

print(LOSS)


fig = plt.figure(facecolor='skyblue')
ax = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8])

ax.scatter(dataset.X, dataset.Y)
x = torch.arange(-2, 2, 0.01).view(-1, 1)
ax.plot(x, model(x).detach().numpy())

plt.show()
