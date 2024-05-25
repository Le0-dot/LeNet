"""
This module implements a LeNet convolutional neural network
for image classification using PyTorch.
It includes functionalities for training and testing the model,
reading datasets, and saving the model and training history.
"""

import csv
import argparse
from pathlib import Path
from functools import reduce
from typing import Callable, Any

import torch
from torch import Tensor
from torch.nn import (
    Flatten,
    LogSoftmax,
    MaxPool2d,
    Module,
    Conv2d,
    Linear,
    NLLLoss,
    ReLU,
)
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.folder import DatasetFolder, pil_loader
from torchvision.transforms import Resize
from torchvision.transforms.functional import convert_image_dtype, pil_to_tensor


class LeNet(Module):
    """
    LeNet convolutional neural network for image classification.

    Attributes:
        conv1 (Conv2d): First convolutional layer.
        relu1 (ReLU): Activation function after the first convolutional layer.
        pool1 (MaxPool2d): Pooling layer after the first activation.
        conv2 (Conv2d): Second convolutional layer.
        relu2 (ReLU): Activation function after the second convolutional layer.
        pool2 (MaxPool2d): Pooling layer after the second activation.
        flatten (Flatten): Layer to flatten the input for fully connected layers.
        fc1 (Linear): First fully connected layer.
        relu3 (ReLU): Activation function after the first fully connected layer.
        fc2 (Linear): Second fully connected layer.
        softmax (LogSoftmax): LogSoftmax activation function for the output layer.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, size: int) -> None:
        super().__init__()
        self.conv1 = Conv2d(3, 20, 5)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(2, 2)

        self.conv2 = Conv2d(20, 50, 5)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(2, 2)

        self.flatten = Flatten()

        self.fc1 = Linear(50 * ((size - 12) // 4) ** 2, 500)
        self.relu3 = ReLU()

        self.fc2 = Linear(500, 150)
        self.softmax = LogSoftmax(dim=1)

    def forward(self, data: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            data (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the network.
        """
        return reduce(lambda x, m: m(x), self.children(), data)


def get_device() -> torch.device:
    """
    Get the available device for computation (CUDA if available, else CPU).

    Returns:
        torch.device: Device to be used for computation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def read_dataset(
    path: Path, transforms: list[Callable[[Tensor], Tensor]]
) -> DatasetFolder:
    """
    Read and transform dataset from the given path.

    Args:
        path (Path): Path to the dataset.
        transforms (list[Callable[[Tensor], Tensor]]): List of transformations for the images.

    Returns:
        DatasetFolder: DatasetFolder object containing the dataset.
    """

    def transform(img: Tensor) -> Tensor:
        return reduce(lambda x, f: f(x), transforms, img)

    def not_svg(path: str) -> bool:
        return not path.endswith(".svg")

    return DatasetFolder(path, pil_loader, transform=transform, is_valid_file=not_svg)


def train(
    model: LeNet,
    dataset: DatasetFolder,
    loss_function: Callable[[Any, Any], Any],
    device: torch.device = get_device(),
) -> tuple[float, float]:
    """
    Train the model on the given dataset.

    Args:
        model (LeNet): The model to be trained.
        dataset (DatasetFolder): The dataset to train on.
        loss_function (Callable[[Any, Any], Any]): The loss function.
        device (torch.device): Device to be used for computation.

    Returns:
        tuple[float, float]: Training loss and accuracy.
    """
    train_loss = torch.tensor(0.0).to(device)
    train_correct = 0

    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    optimizer = Adam(model.parameters())

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        prediction = model(x)
        loss = loss_function(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
        train_correct += (prediction.argmax(1) == y).type(torch.int).sum().item()

    return train_loss.item(), train_correct / len(dataset)


def test(
    model: LeNet,
    dataset: DatasetFolder,
    loss_function: Callable[[Any, Any], Any],
    device: torch.device = get_device(),
) -> tuple[float, float]:
    """
    Test the model on the given dataset.

    Args:
        model (LeNet): The model to be tested.
        dataset (DatasetFolder): The dataset to test on.
        loss_function (Callable[[Any, Any], Any]): The loss function.
        device (torch.device): Device to be used for computation.

    Returns:
        tuple[float, float]: Test loss and accuracy.
    """
    test_loss = torch.tensor(0.0).to(device)
    test_correct = 0

    test_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    with torch.no_grad():
        model.eval()

        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            loss = loss_function(prediction, y)

            test_loss += loss
            test_correct += (prediction.argmax(1) == y).type(torch.int).sum().item()

    return test_loss.item(), test_correct / len(dataset)


def save(model: LeNet, history: dict[str, list[float]]) -> None:
    """
    Save the model and training history to disk.

    Args:
        model (LeNet): The model to be saved.
        history (dict[str, list[float]]): Training history.
    """
    torch.save(model, "model.pt")
    with open("history.csv", mode="w", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(history.keys())
        writer.writerows(zip(*list(history.values())))


def get_args() -> argparse.Namespace:
    """
    Parse and return command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--size", type=int, default=64)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Main function to train and test the model based on command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    model = LeNet(args.size).to(get_device())

    transforms = [
        pil_to_tensor,
        Resize((args.size, args.size)),
        lambda x: x[:3],
        convert_image_dtype,
    ]

    train_dataset = read_dataset(Path() / "pokemons" / "train", transforms)
    test_dataset = read_dataset(Path() / "pokemons" / "test", transforms)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        train_loss, train_accuracy = train(model, train_dataset, NLLLoss())
        test_loss, test_accuracy = test(model, test_dataset, NLLLoss())

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(test_loss)
        history["validation_accuracy"].append(test_accuracy)

    save(model, history)


if __name__ == "__main__":
    main(get_args())
