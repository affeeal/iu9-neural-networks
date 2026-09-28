"""Train one model/optimizer pair; datasets are downloaded only with --download."""

import argparse
import math
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MODELS = ("lenet5", "vgg16", "resnet34")
OPTIMIZERS = ("sgd", "adadelta", "nag", "adam")


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


def create_model(name):
    if name == "lenet5":
        return LeNet5(num_classes=10)
    if name == "vgg16":
        return models.vgg16(weights=None, num_classes=10, dropout=0.5)
    if name == "resnet34":
        return models.resnet34(weights=None, num_classes=10)
    raise ValueError(f"Unknown model: {name}")


def create_optimizer(name, parameters, learning_rate):
    if name == "sgd":
        return optim.SGD(parameters, lr=learning_rate)
    if name == "adadelta":
        return optim.Adadelta(parameters, lr=learning_rate)
    if name == "nag":
        return optim.SGD(parameters, lr=learning_rate, momentum=0.9, nesterov=True)
    if name == "adam":
        return optim.Adam(parameters, lr=learning_rate)
    raise ValueError(f"Unknown optimizer: {name}")


def train(n_epochs, optimizer, model, loss_fn, train_loader, device):
    """Train with a mean-reduced loss, returning sample-weighted epoch losses."""
    if n_epochs <= 0:
        raise ValueError("Epoch count must be positive")
    losses = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        loss_sum = 0.0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += labels.size(0)
            loss_sum += loss.item() * labels.size(0)
        if total == 0:
            raise ValueError("Empty training loader")
        losses.append(loss_sum / total)
        print(f"Epoch {epoch}/{n_epochs}, training loss: {losses[-1]:.6f}")
    return losses


def accuracy(model, loader, device):
    # Preserve per-module modes, including intentionally frozen BatchNorm layers.
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    correct = total = 0
    try:
        with torch.inference_mode():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images).argmax(dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        if total == 0:
            raise ValueError("Empty evaluation loader")
        return correct / total
    finally:
        for module, training in modes:
            module.training = training


def load_datasets(model_name, directory, download=False):
    if model_name == "lenet5":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        dataset = datasets.MNIST
    elif model_name in ("vgg16", "resnet34"):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
        ])
        dataset = datasets.CIFAR10
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return (
        dataset(directory, train=True, download=download, transform=transform),
        dataset(directory, train=False, download=download, transform=transform),
    )


def run_experiment(model_name, optimizer_name, training, testing, *,
                   epochs, batch_size, learning_rate, seed, device):
    # Each optimizer starts from the same weights and data-order RNG state.
    torch.manual_seed(seed)
    model = create_model(model_name).to(device)
    optimizer = create_optimizer(optimizer_name, model.parameters(), learning_rate)
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(testing, batch_size=batch_size, shuffle=False)
    print(f"{model_name}, {optimizer_name}, device={device}, seed={seed}")
    losses = train(epochs, optimizer, model, nn.CrossEntropyLoss(), train_loader, device)
    scores = {
        "train": accuracy(model, DataLoader(training, batch_size=batch_size), device),
        "test": accuracy(model, test_loader, device),
    }
    print(", ".join(f"{name} accuracy: {value:.3f}" for name, value in scores.items()))
    return losses, scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="lenet5")
    parser.add_argument("--optimizer", choices=(*OPTIMIZERS, "all"), default="sgd")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be between 0 and 2**63 - 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu or install a CUDA build")
    try:
        training, testing = load_datasets(args.model, args.data_dir, args.download)
        for optimizer in OPTIMIZERS if args.optimizer == "all" else (args.optimizer,):
            run_experiment(args.model, optimizer, training, testing,
                           epochs=args.epochs, batch_size=args.batch_size,
                           learning_rate=args.learning_rate, seed=args.seed,
                           device=torch.device(args.device))
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
