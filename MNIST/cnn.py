import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_LOADER = torch.utils.data.DataLoader(
    datasets.MNIST('../data/', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=64, shuffle=True
)
TEST_LOADER = torch.utils.data.DataLoader(
    datasets.MNIST('../data/', train=False, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=64, shuffle=True
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Args:
            x(tensor NCHW format)
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


model = Net().to(DEVICE)
optimizer = optim.Adam(model.parameters())


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(TRAIN_LOADER):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print("Training epoch: {} [{}/{} ({:0f}%]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(TRAIN_LOADER.dataset),
                100. * batch_idx / len(TRAIN_LOADER), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in TEST_LOADER:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False)

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(TEST_LOADER.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuraccy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(TEST_LOADER.dataset),
        100. * correct / len(TEST_LOADER.dataset)))


def main(epochs):
    print(DEVICE)
    for epoch in range(1, epochs + 1):
        train(epoch)
        test()
