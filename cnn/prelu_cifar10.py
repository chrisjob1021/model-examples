# Example CNN training on CIFAR10 with optional PReLU activation
# Inspired by "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
# (He et al., 2015) - https://arxiv.org/abs/1502.01852

import argparse
import os
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def get_dataloaders(batch_size: int = 128) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Return train and test dataloaders for CIFAR10."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


class SimpleCNN(nn.Module):
    """Small CNN with optional PReLU activation."""

    def __init__(self, use_prelu: bool = False) -> None:
        super().__init__()
        activation: nn.Module
        if use_prelu:
            activation = nn.PReLU
        else:
            activation = nn.ReLU

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            activation(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            activation(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            activation(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, 256),
            activation(),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = self.classifier(x)
        return x


def train(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
    loss = running_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return loss, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR10 CNN with optional PReLU")
    parser.add_argument("--prelu", action="store_true", help="Use PReLU activation instead of ReLU")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader, testloader = get_dataloaders(batch_size=args.batch_size)
    model = SimpleCNN(use_prelu=args.prelu).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        print(f"Epoch {epoch}: train loss={train_loss:.4f} test loss={test_loss:.4f} test acc={test_acc:.4f}")

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "cnn_prelu.pt" if args.prelu else "cnn_relu.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
