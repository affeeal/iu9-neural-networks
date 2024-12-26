import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional

from torchvision import datasets, transforms

DATA_PATH = 'data/'


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = functional.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = functional.max_pool2d(torch.relu(self.conv2(out)), 2)
        out = out.view(-1, 16 * 5 * 5)
        out = torch.tanh(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        out = self.fc3(out)
        return out


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch,
            loss_train / len(train_loader)))


def accuracy(model, train_loader, test_loader):
    accdict = {}
    for name, loader in [("train", train_loader), ("test", test_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.3f}".format(name, correct / total))
        accdict[name] = correct / total
    return accdict


if __name__ == '__main__':
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f"Training on device {device}.")

    mnist_train = datasets.MNIST(
        DATA_PATH, train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(
        DATA_PATH, train=False, download=True, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=100, shuffle=True)

    model = LeNet().to(device=device)
    loss_fn = nn.CrossEntropyLoss()

    for name, optimizer in [
        ('SGD', optim.SGD(model.parameters(), lr=1e-2)),
        ('Adadelta', optim.Adadelta(model.parameters(), lr=1e-2)),
        ('NAG', optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)),
            ('Adam', optim.Adam(model.parameters(), lr=1e-3))]:
        print(f'Optimization {name}.')
        training_loop(n_epochs=20, optimizer=optimizer, model=model,
                      loss_fn=loss_fn, train_loader=train_loader)
        accuracy(model, train_loader, test_loader)
