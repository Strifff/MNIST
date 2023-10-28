import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from test_installation import test_pytorch, show_random_img
from neural_network.network import SimpleNN, evaluate_batch
from utils.data_utils import skew, rotate, pad_image, image_to_tensor, blur, scale_down_image, shadow
from utils.plot_utils import plot_grid


def install_test():
    test_pytorch()


def train_simple_nn():
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

    # Initialize the model, loss function, and optimizer
    model = SimpleNN(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

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

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
        eval_data = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
        data_iterator = iter(eval_data)
        batch = next(data_iterator)
        evaluate_batch(model, batch)

    # Save the trained model
    # torch.save(model.state_dict(), "mnist_model.pth")


def augment_data():
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(
        root="./mnist_data", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(mnist_dataset, batch_size=1, shuffle=True)
    original_image, original_label = next(iter(dataloader))

    data = []

    # Original image
    original_label = str(original_label.numpy()[0])
    label = original_label + " (original)"
    data += [(original_image, label)]

    # Padded image
    padded_image = pad_image(original_image.numpy()[0], 14)
    padded_image = scale_down_image(padded_image, 28, 28)
    padded_image = image_to_tensor(padded_image)
    label = original_label + " (padded)"
    data += [(padded_image, label)]

    # Skew image
    skewed_image = skew(original_image.numpy()[0], 0.5)
    skewed_image = image_to_tensor(skewed_image)
    label = original_label + " (skewed)"
    data += [(skewed_image, label)]

    # Rotate image
    rotated_image = rotate(original_image.numpy()[0])
    rotated_image = image_to_tensor(rotated_image)
    label = original_label + " (rotated)"
    data += [(rotated_image, label)]

    # Blur image
    blurred_image = blur(original_image.numpy()[0])
    blurred_image = image_to_tensor(blurred_image)
    label = original_label + " (blurred)"
    data += [(blurred_image, label)]
    
    # Shadow
    shadow_image = shadow(original_image.numpy()[0])
    shadow_image = image_to_tensor(shadow_image)
    label = original_label + " (shadow)"
    data += [(shadow_image, label)]
    
    
    # Plot all
    plot_grid(data)


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
        "--data_augmentation", action="store_true", help="Augment the data"
    )

    args = parser.parse_args()

    if args.test_installation:
        install_test()

    if args.show_random_img:
        path = "data/archive/train-images-idx3-ubyte"
        grid_size = 4
        show_random_img(path, grid_size)

    if args.train_simple_nn:
        train_simple_nn()

    if args.data_augmentation:
        augment_data()


if __name__ == "__main__":
    main()
