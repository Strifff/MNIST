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
from test_installation import test_pytorch, show_random_img
from neural_network.network import SimpleNN, SimpleCNN, evaluate_batch
from utils.data_utils import (
    skew,
    rotate,
    pad_image,
    image_to_tensor,
    blur,
    scale_down_image,
    augment_image,
    AugmentedDataset,
    DS2,
)
from utils.plot_utils import plot_grid, CustomProgressBar


def install_test():
    test_pytorch()


def train_simple_NN():
    # Hyperparameters
    input_size = 28 * 28  # MNIST images are 28x28 pixels
    hidden_size = 128  # Number of neurons in the hidden layer
    num_classes = 10  # MNIST has 10 classes (0-9)
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 1000

    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(
        root="./mnist_data", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

    eval_dataset = datasets.MNIST(
        root="./mnist_data", train=False, transform=transform, download=True
    )

    # Initialize the model, loss function, and optimizer
    model = SimpleNN(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            # Forward pass
            print("images simple NN: ", images.shape, images.dtype)
            outputs = model(images)
            print("outputs simple NN: ", outputs.shape, outputs.dtype)
            loss = criterion(outputs, labels)
            print("loss simple NN: ", loss.shape, loss.dtype)
            print(loss)
            exit()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
        eval_data = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
        data_iterator = iter(eval_data)
        batch = next(data_iterator)
        evaluate_batch(model, batch)

    # Save the trained model
    # torch.save(model.state_dict(), "mnist_model.pth")


def train_simple_CNN():
    # Hyperparameters
    input_size = 28 * 28  # MNIST images are 28x28 pixels
    hidden_size = 128  # Number of neurons in the hidden layer
    num_classes = 10  # MNIST has 10 classes (0-9)
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 1000

    # Initialize dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(
        root="./mnist_data", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)

    eval_dataset = datasets.MNIST(
        root="./mnist_data", train=False, transform=transform, download=True
    )

    # Initialize the model, loss function, and optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            # Forward pass
            outputs = model(images, input_size, hidden_size, num_classes)
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
        eval_data = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
        data_iterator = iter(eval_data)
        batch = next(data_iterator)
        evaluate_batch(model, batch)


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
    transform = transforms.Compose([transforms.ToTensor()])
    augmented_dataset = AugmentedDataset(transform=transform)
    aug_dl = DataLoader(augmented_dataset, batch_size=10, shuffle=True)
    for images, labels in aug_dl:
        for image, label in zip(images, labels):
            plt.imshow(image.reshape(28, 28), cmap="gray")
            plt.title(f"Label: {label.item()}")
            plt.show()
        break


def main():
    parser = argparse.ArgumentParser(description="MNIST refresher")

    parser.add_argument(
        "--test_installation",
        action="store_true",
        help="Train the custom Transformer model",
    )

    parser.add_argument(
        "--show_random_img",
        action="store_true",
        help="Show a random image from the MNIST dataset",
    )

    parser.add_argument(
        "--train_simple_nn",
        action="store_true",
        help="Train the network",
    )

    parser.add_argument(
        "--train_simple_cnn",
        action="store_true",
        help="Train the network",
    )

    parser.add_argument(
        "--clear_augmented_data",
        action="store_true",
        help="Clear augmented data",
    )

    parser.add_argument(
        "--misc",
        action="store_true",
        help="Misceleanous stuff",
    )

    parser.add_argument(
        "--data_augmentation", action="store_true", help="Augment the data"
    )

    args = parser.parse_args()

    if args.test_installation:
        install_test()

    if args.show_random_img:
        path = "data/archive/train-images-idx3-ubyte"
        show_random_img(path)

    if args.train_simple_nn:
        train_simple_NN()

    if args.train_simple_cnn:
        train_simple_CNN()

    if args.data_augmentation:
        augment_data()

    if args.clear_augmented_data:
        clear_augmented_data()

    if args.misc:
        misc()


if __name__ == "__main__":
    main()
