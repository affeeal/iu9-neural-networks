import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional

from torchvision import datasets, transforms

DATA_PATH = '../datasets/'
BATCH_SIZE = 100
MOMENTUM = 0.9
EPOCHS = 20


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


class VGG16(nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.conv11 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv31 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc1_dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc2_dropout = nn.Dropout(p=dropout_p)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        out = torch.relu(self.conv11(x))     # 3 x 32 x 32   -> 128 x 32 x 32
        out = torch.relu(self.conv12(out))   # 128 x 32 x 32 -> 128 x32 x 32
        out = functional.max_pool2d(out, 2)  # 128 x 32 x 32 -> 128 x 16 x 16
        out = torch.relu(self.conv21(out))   # 128 x 16 x 16 -> 256 x 16 x 16
        out = torch.relu(self.conv22(out))   # 256 x 16 x 16 -> 256 x 16 x 16
        out = functional.max_pool2d(out, 2)  # 256 x 16 x 16 -> 256 x 8 x 8
        out = torch.relu(self.conv31(out))   # 256 x 8 x 8   -> 512 x 8 x 8
        out = torch.relu(self.conv32(out))   # 512 x 8 x 8   -> 512 x 8 x 8
        out = functional.max_pool2d(out, 2)  # 512 x 8 x 8   -> 512 x 4 x 4
        out = out.view(-1, 512 * 4 * 4)
        out = torch.relu(self.fc1(out))      # 512 x 4 x 4 -> 1024
        out = self.fc1_dropout(out)
        out = torch.relu(self.fc2(out))      # 1024 -> 1024
        out = self.fc2_dropout(out)
        out = self.fc3(out)                  # 1024 -> 10
        return out


def train(n_epochs, optimizer, model, loss_fn, train_loader):
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


def calculate_accuracy(model, train_loader, test_loader):
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


device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
loss_fn = nn.CrossEntropyLoss()

mnist_train = datasets.MNIST(
    DATA_PATH, train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(
    DATA_PATH, train=False, download=True, transform=transforms.ToTensor())

cifar10_train = datasets.CIFAR10(
    DATA_PATH, train=True, download=True, transform=transforms.ToTensor())
cifar10_test = datasets.CIFAR10(
    DATA_PATH, train=False, download=True, transform=transforms.ToTensor())


# LeNet, MNIST
model = LeNet().to(device=device)
train_loader = torch.utils.data.DataLoader(
    mnist_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    mnist_test, batch_size=BATCH_SIZE, shuffle=True)

# SGD
optimizer = optim.SGD(model.parameters(), lr=1e-2)
train(n_epochs=EPOCHS, optimizer=optimizer, model=model,
      loss_fn=loss_fn, train_loader=train_loader)
calculate_accuracy(model, train_loader, test_loader)

# Adadelta
optimizer = optim.Adadelta(model.parameters(), lr=1e-2)
train(n_epochs=EPOCHS, optimizer=optimizer, model=model,
      loss_fn=loss_fn, train_loader=train_loader)
calculate_accuracy(model, train_loader, test_loader)

# NAG
optimizer = optim.SGD(model.parameters(), lr=1e-2,
                      momentum=MOMENTUM, nesterov=True)
train(n_epochs=EPOCHS, optimizer=optimizer, model=model,
      loss_fn=loss_fn, train_loader=train_loader)
calculate_accuracy(model, train_loader, test_loader)

# Adam
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train(n_epochs=EPOCHS, optimizer=optimizer, model=model,
      loss_fn=loss_fn, train_loader=train_loader)
calculate_accuracy(model, train_loader, test_loader)


# VGG16, CIFAR10
model = VGG16(dropout_p=0.4).to(device=device)
train_loader = torch.utils.data.DataLoader(
    cifar10_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    cifar10_test, batch_size=BATCH_SIZE, shuffle=True)

# SGD
optimizer = optim.SGD(model.parameters(), lr=1e-1)
train(n_epochs=EPOCHS, optimizer=optimizer, model=model,
      loss_fn=loss_fn, train_loader=train_loader)
calculate_accuracy(model, train_loader, test_loader)

# Adadelta
optimizer = optim.Adadelta(model.parameters(), lr=1e-1)
train(n_epochs=EPOCHS, optimizer=optimizer, model=model,
      loss_fn=loss_fn, train_loader=train_loader)
calculate_accuracy(model, train_loader, test_loader)

# NAG
optimizer = optim.SGD(model.parameters(), lr=1e-1,
                      momentum=MOMENTUM, nesterov=True)
train(n_epochs=EPOCHS, optimizer=optimizer, model=model,
      loss_fn=loss_fn, train_loader=train_loader)
calculate_accuracy(model, train_loader, test_loader)

# Adam
optimizer = optim.Adam(model.parameters(), lr=1e-1)
train(n_epochs=EPOCHS, optimizer=optimizer, model=model,
      loss_fn=loss_fn, train_loader=train_loader)
calculate_accuracy(model, train_loader, test_loader)
