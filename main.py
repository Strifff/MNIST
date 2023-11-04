import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from test_installation import test_pytorch, show_random_img
from neural_network.network import (
    SimpleNN,
    SimpleCNN,
    DynamicCNN,
    evaluate_batch,
    save_hyperparametes,
    load_hyperparameters,
)

from utils.data_utils import augment_image, AugmentedDataset
from utils.plot_utils import plot_grid, CustomProgressBar


def test_installation():
    test_pytorch()


def train_simple_NN():
    # Hyperparameters
    input_size = 28 * 28  # MNIST images are 28x28 pixels
    hidden_size = 128  # Number of neurons in the hidden layer
    num_classes = 10  # MNIST has 10 classes (0-9)
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 1000

    # Initialize the model, loss function, and optimizer
    model = SimpleNN(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(
        root="./mnist_data", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

    eval_dataset = datasets.MNIST(
        root="./mnist_data", train=False, transform=transform, download=True
    )

    # Training loop
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        eval_data = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
        data_iterator = iter(eval_data)
        batch = next(data_iterator)
        evaluate_batch(model, batch)

    # Save the trained model
    # torch.save(model.state_dict(), "mnist_model.pth")


def train_simple_CNN():
    # Hyperparameters
    input_size = 28 * 28  # MNIST images are 28x28 pixels
    num_classes = 10  # MNIST has 10 classes (0-9)
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 100
    num_batches = 1000

    # Initialize the model, loss function, and optimizer
    model = SimpleCNN(input_channels=1, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Initialize dataloader
    transform = transforms.Compose([transforms.ToTensor()])

    augmented_dataset = AugmentedDataset(transform=transform)
    dataloader = DataLoader(augmented_dataset, batch_size=64, shuffle=True)

    eval_dataset = datasets.MNIST(
        root="./mnist_data", train=False, transform=transform, download=True
    )
    eval_data = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    process = CustomProgressBar(total_items=num_epochs, desc="Training")
    # Training loop
    for epoch in range(num_epochs):
        i = 0
        for images, labels in dataloader:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Too large augmented dataset
            i += 1
            if i == num_batches:
                break
        process.update_progress()
        evaluate_batch(model, eval_data, epoch)


def train_dynamic_CNN():
    hyper_path = "neural_network/hyperparameters.json"
    if os.path.exists(hyper_path):
        hyperparameters = load_hyperparameters(hyper_path)
    else:
        # Init hyperparameters
        hyperparameters = {
            "batch_size": 64,
            "learning_rate": 0.001,
            "input_channels": 1,
            "input_hidden_channels": 32,
            "input_kernel_size": 3,
            "input_stride": 1,
            "input_pool_kernel_size": 2,
            "input_pool_stride": 2,
            "num_hidden_layers": 2,
            "hidden_channels": [32, 64],
            "kernel_sizes": [3, 3],
            "strides": [1, 1],
            "pool_kernel_sizes": [2, 2],
            "pool_strides": [2, 2],
            "fc_out_features": 128,
            "num_classes": 10,
        }
        save_hyperparametes(hyperparameters, hyper_path)

    # Initialize the model, loss function, and optimizer
    model = DynamicCNN(
        input_channels=hyperparameters["input_channels"],
        input_hidden_channels=hyperparameters["input_hidden_channels"],
        input_kernel_size=hyperparameters["input_kernel_size"],
        input_stride=hyperparameters["input_stride"],
        input_pool_kernel_size=hyperparameters["input_pool_kernel_size"],
        input_pool_stride=hyperparameters["input_pool_stride"],
        num_hidden_layers=hyperparameters["num_hidden_layers"],
        hidden_channels=hyperparameters["hidden_channels"],
        kernel_sizes=hyperparameters["kernel_sizes"],
        strides=hyperparameters["strides"],
        pool_kernel_sizes=hyperparameters["pool_kernel_sizes"],
        pool_strides=hyperparameters["pool_strides"],
        fc_out_features=hyperparameters["fc_out_features"],
        num_classes=hyperparameters["num_classes"],
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Initialize dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    aug_data = AugmentedDataset(transform=transform)
    dataloader = DataLoader(aug_data, batch_size=batch_size, shuffle=True)

    eval_dataset = datasets.MNIST(
        root="./mnist_data", train=False, transform=transform, download=True
    )
    eval_data = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)


def clear_augmented_data():
    custom_dataset = AugmentedDataset()
    custom_dataset.clear_dataset()


def augment_data():
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_dataset = datasets.MNIST(
        root="./data/mnist_data", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(mnist_dataset, batch_size=600, shuffle=True)

    augmented_dataset = AugmentedDataset(transform=transform)
    augmented_dataset.clear_dataset()

    progress = CustomProgressBar(total_items=len(dataloader), desc="Augmenting data")

    for images, labels in dataloader:
        progress.update_progress()
        for image, label in zip(images, labels):
            augmented = augment_image(image.numpy()[0])
            label = torch.tensor(label.item(), dtype=torch.int64).reshape(1)
            for im in augmented:
                augmented_dataset.add_image_label(im, label)
    augmented_dataset.save_to_files()


def misc():
    d_cnn = DynamicCNN()


def main():
    parser = argparse.ArgumentParser(description="MNIST refresher")

    actions = {
        "test_installation": test_installation,
        "train_simple_nn": train_simple_NN,
        "train_simple_cnn": train_simple_CNN,
        "train_dynamic_cnn": train_dynamic_CNN,
        "data_augmentation": augment_data,
        "clear_augmented_data": clear_augmented_data,
        "misc": misc,
    }

    for action, function in actions.items():
        parser.add_argument(f"--{action}", action="store_true", help=function.__doc__)

    args = parser.parse_args()

    for action, function in actions.items():
        if getattr(args, action):
            function()


if __name__ == "__main__":
    main()
