import torch
from torch import nn

torch.manual_seed(1)


class Data(torch.utils.data.Dataset):
    """Some Information about Data"""

    def __init__(self):
        self.x = torch.zeros(20, 2)
        self.x[:, 0] = torch.arange(-1, 1, 0.1)
        self.x[:, 1] = torch.arange(-1, 1, 0.1)
        self.w = torch.tensor([[1.0, -1.0], [1.0, 3.0]])
        self.b = torch.tensor([[-1.0, 1.0]])
        self.f = torch.mm(self.x, self.w) + self.b

        # make random noize
        self.y = self.f + 0.001 * torch.randn(self.x.shape[0], 1)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


class LinearRegression(nn.Module):
    """Some Information about LinearRegression"""

    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        yhat = self.linear(x)
        return yhat


model = LinearRegression(2, 2)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

criterion = nn.MSELoss()

dataset = Data()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=5)

LOSS = []
epochs = 100

for epoch in range(epochs):
    for x, y in dataloader:
        yhat = model(x)
        loss = criterion(yhat, y)
        LOSS.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print(LOSS)
