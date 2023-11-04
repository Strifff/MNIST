import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate input size for the fully connected layer after convolution
        self.fc_input_size = 32 * 7 * 7  # Assuming input image size of 28x28

        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x  # torch.softmax(x, dim=1)


class DynamicCNN(nn.Module):
    def __init__(
        self,
        input_channels,
        input_hidden_channels,
        input_kernel_size,
        input_stride,
        input_pool_kernel_size,
        input_pool_stride,
        num_hidden_layers,
        hidden_channels,
        kernel_sizes,
        strides,
        pool_kernel_sizes,
        pool_strides,
        fc_out_features,
        num_classes,
    ):
        super(DynamicCNN, self).__init()
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_strides = conv_strides
        self.pool_kernel_sizes = pool_kernel_sizes
        self.pool_strides = pool_strides
        self.layers = self._build_layers(
            num_hidden_layers,
            in_channels,
            hidden_channels,
            fc_out_features,
            num_classes,
        )

    def _build_layers(
        self,
        num_hidden_layers,
        in_channels,
        hidden_channels,
        fc_out_features,
        num_classes,
    ):
        layers = []

        # Add the input layer
        layers.append(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=self.conv_kernel_sizes[0],
                stride=self.conv_strides[0],
                padding=1,
            )
        )
        layers.append(nn.ReLU())
        layers.append(
            nn.MaxPool2d(
                kernel_size=self.pool_kernel_sizes[0], stride=self.pool_strides[0]
            )
        )

        # Add hidden layers
        for i in range(1, num_hidden_layers + 1):
            layers.append(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=self.conv_kernel_sizes[i],
                    stride=self.conv_strides[i],
                    padding=1,
                )
            )
            layers.append(nn.ReLU())
            layers.append(
                nn.MaxPool2d(
                    kernel_size=self.pool_kernel_sizes[i], stride=self.pool_strides[i]
                )
            )

        # Flatten and add fully connected layers
        layers.append(nn.Flatten())
        layers.append(nn.Linear(hidden_channels * 7 * 7, fc_out_features))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(fc_out_features, num_classes))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# Example usage:
# in_channels = 1  # For grayscale images
# num_classes = 10  # MNIST has 10 classes
# hidden_channels = 32
# fc_out_features = 128
# conv_kernel_sizes = [3, 3, 3]  # List of kernel sizes for each convolutional layer
# conv_strides = [1, 1, 1]  # List of strides for each convolutional layer
# pool_kernel_sizes = [2, 2, 2]  # List of kernel sizes for each pooling layer
# pool_strides = [2, 2, 2]  # List of strides for each pooling layer


def save_hyperparametes(hyperparameters, path):
    with open(path, "w") as f:
        json.dump(hyperparameters, f)


def load_hyperparameters(path):
    with open(path, "r") as f:
        hyperparameters = json.load(f)
    return hyperparameters


def evaluate_batch(model, eval_data, epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in eval_data:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100
    print(
        f"Epoch {epoch}:\tAccuracy = {'{:.2f}'.format(np.round(accuracy,2))}%\t{correct}/{total}"
    )
