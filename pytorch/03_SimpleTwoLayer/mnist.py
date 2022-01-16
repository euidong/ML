from random import shuffle
import torch
from torch import nn
import torchvision.datasets as dset
from torchvision import transforms


class SimpleTwoLayer(nn.Module):
    """Some Information about SimpleTwoLayer"""

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleTwoLayer, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.l1(x))
        x = self.l2(x)
        return x


criterion = nn.CrossEntropyLoss()

model = SimpleTwoLayer(28 * 28, 100, 10)

trainset = dset.MNIST(root='./data', train=True,
                      download=False, transform=transforms.ToTensor())
validationset = dset.MNIST(root='./data', train=False,
                           download=False, transform=transforms.ToTensor())

traindataloader = torch.utils.data.DataLoader(
    trainset, batch_size=2000, shuffle=True)
validationdataloader = torch.utils.data.DataLoader(
    validationset, batch_size=5000, shuffle=False)

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 100
LOSS = []

for _ in range(epochs):
    epoch_loss = 0
    for x, y in traindataloader:
        optimizer.zero_grad()
        yhat = model(x.view(-1, 28*28))
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    LOSS.append(epoch_loss)

print(LOSS)
correct = 0
for x, y in validationdataloader:
    z = model(x.reshape(-1, 28 * 28))
    _, yhat = torch.max(z, 1)
    correct += (yhat == y).sum().item()

print(correct/10000)
