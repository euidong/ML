import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim


class Dataset(torch.utils.data.Dataset):
    """Some Information about Dataset"""

    def __init__(self):
        super(Dataset, self).__init__()
        torch.random.manual_seed(1)
        x1_1 = ((torch.rand(10) - 0.5) / 5).view(-1, 1)
        x1_2 = ((torch.rand(10) - 0.5) / 5).view(-1, 1)
        x1 = torch.cat([x1_1, x1_2], dim=1)
        y1 = torch.zeros(x1.shape[0])

        x2_1 = ((torch.rand(10) - 0.5) / 5).view(-1, 1)
        x2_2 = ((torch.rand(10) - 0.5) / 5 + 1).view(-1, 1)
        x2 = torch.cat([x2_1, x2_2], dim=1)
        y2 = torch.ones(x2.shape[0])

        x3_1 = ((torch.rand(10) - 0.5) / 5 + 1).view(-1, 1)
        x3_2 = ((torch.rand(10) - 0.5) / 5).view(-1, 1)
        x3 = torch.cat([x3_1, x3_2], dim=1)
        y3 = torch.ones(x3.shape[0])

        x4_1 = ((torch.rand(10) - 0.5) / 5 + 1).view(-1, 1)
        x4_2 = ((torch.rand(10) - 0.5) / 5 + 1).view(-1, 1)
        x4 = torch.cat([x4_1, x4_2], dim=1)
        y4 = torch.zeros(x4.shape[0])

        self.x = torch.cat([x1, x2, x3, x4])
        self.y = torch.cat([y1, y2, y3, y4])

        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


class SimpleTwoLayer(nn.Module):
    """Some Information about SimpleTwoLayer"""

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleTwoLayer, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        z1 = self.l1(x)
        o1 = torch.sigmoid(z1)
        z2 = self.l2(o1)
        yhat = torch.sigmoid(z2)
        return yhat


def criterion(yhat, y):
    out = -1 * torch.mean(y * torch.log(yhat) +
                          (1 - y) * torch.log(1 - yhat))
    return out


dataset = Dataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
model = SimpleTwoLayer(2, 2, 1)

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


fig = plt.figure(facecolor='skyblue')
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)


weight = model.l1.weight.detach().numpy()
bias = model.l1.bias.detach().numpy()

# x1 * w1 + x2 * w2 + b = 0
# x2 = - (x1 * w1 + b) / w2


def line(w1, w2, b):
    x = np.arange(-0.2, 1.4, 0.1)
    y = -1 * (x * w1 + b) / w2
    return x, y


x1, y1 = line(weight[0][0], weight[0][1], bias[0])
x2, y2 = line(weight[1][0], weight[1][1], bias[1])

ax1.plot(x1, y1)
ax1.plot(x2, y2)
ax1.fill_between(x1, y1=y1, y2=y2, color='lightpink', alpha=0.5)
ax1.fill_between(x1, y1=2, y2=y2, color='skyblue', alpha=0.5)
ax1.fill_between(x1, y1=y1, y2=-1, color='skyblue', alpha=0.5)
ax1.scatter(dataset.x[:, 0], dataset.x[:, 1], c=dataset.y)

ax2.plot(np.arange(0, len(LOSS), 1), LOSS)


plt.show()
