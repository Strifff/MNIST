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
from neural_network.network import SimpleNN, evaluate_batch
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


def augment_data_tester():
    print("TESTER")
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(
        root="./data/mnist_data", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(mnist_dataset, batch_size=1, shuffle=True)
    original_image, original_label = next(iter(dataloader))
    # print("image: ", original_image.shape, original_image.dtype)
    # print("label: ", original_label.shape, original_label.dtype)

    dataloader = DataLoader(mnist_dataset, batch_size=10, shuffle=True)
    # print("dl std: ", next(iter(dataloader))[0].shape, next(iter(dataloader))[1].shape)

    print("original: ", original_image.shape, original_label.shape)
    print("original: ", original_label.shape, original_label.dtype)
    augmented = augment_image(original_image.numpy()[0])

    im_label = []

    for im in augmented:
        im_label += [(im, original_label)]

    plot_grid(im_label)
    return
    # my_aug_dataset = AugmentedDataset()

    # my_aug_dataset.add_image_label(original_image.numpy()[0], original_label.numpy()[0])

    # custom_dataset = AugmentedDataset(transform=transform)
    # custom_dataset.clear_dataset()
    # for image, label in im_label:
    # custom_dataset.add_image_label(image, label)
    # pass

    # dataloader = DataLoader(custom_dataset, batch_size=100, shuffle=True)

    # print("dl aug: ", next(iter(dataloader))[0].shape, next(iter(dataloader))[1].shape)

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

    # Plot all
    plot_grid(data)
    return

    # Data augmentation
    mnist_dataset = datasets.MNIST(
        root="./data/mnist_data", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(mnist_dataset, batch_size=100, shuffle=True)

    custom_dataset = AugmentedDataset(transform=transform)
    # custom_dataset.clear_dataset()

    for images, labels in dataloader:
        for image, label in zip(images, labels):
            # print("input: ", image.shape, label.shape)
            augmented = augment_image(image.numpy()[0])
            # print("augmented: ", augmented.append)
            label = torch.tensor(label.item(), dtype=torch.int64).reshape(1)
            for im in augmented:
                # image_np = im.numpy()
                custom_dataset.add_image_label(im, label)

    # length = len(custom_dataset)
    # for _ in range(5):
    #    index = int(np.random.uniform(0, length))
    #    entry = custom_dataset[index]
    #    plt.imshow(entry[0].reshape(28,28), cmap="gray")
    #    plt.title(f"Label: {entry[1].item()}")
    #    plt.show()


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


def ds2tester():
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_dataset = datasets.MNIST(
        root="./data/mnist_data", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(mnist_dataset, batch_size=100, shuffle=True)

    image, label = next(iter(dataloader))
    print("original image: ", image.shape, image.dtype)
    print("original label: ", label.shape, label.dtype)

    ds2 = DS2(transform=transform)
    ds2.clear_dataset()

    for images, labels in dataloader:
        for image, label in zip(images, labels):
            augmented = augment_image(image.numpy()[0])
            label = torch.tensor(label.item(), dtype=torch.int64).reshape(1)
            for im in augmented:
                ds2.add_image_label(im, label)

    ds2.save_to_files()

    ds3 = DS2(transform=transform)

    dl3 = DataLoader(ds3, batch_size=10, shuffle=True)

    for images, labels in dl3:
        for image, label in zip(images, labels):
            plt.imshow(image.reshape(28, 28), cmap="gray")
            plt.title(f"MAIN - Label: {label.item()}")
            # print(image.dtype, image)
            plt.show()
        break


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
        train_simple_nn()

    if args.data_augmentation:
        augment_data()

    if args.clear_augmented_data:
        clear_augmented_data()

    if args.misc:
        misc()


if __name__ == "__main__":
    main()
