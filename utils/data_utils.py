import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import numpy as np
import struct
import os
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class AugmentedDataset(Dataset):
    def __init__(
        self,
        image_file="./data/augmented_train_data/train-images-aug-idx3-ubyte",
        label_file="./data/augmented_train_data/train-labels-aug-idx1-ubyte",
        transform=None,
    ):
        self.image_file = image_file
        self.label_file = label_file
        self.images = self.load_images()
        self.labels = self.load_labels()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

    def load_images(self):
        images = []
        try:
            with open(self.image_file, "rb") as file:
                magic, num_images, rows, cols = struct.unpack(">IIII", file.read(16))
                image_data = file.read()

                for i in range(num_images):
                    image = np.frombuffer(
                        image_data[i * rows * cols : (i + 1) * rows * cols],
                        dtype=np.uint8,
                    )

                    image = np.reshape(image, (rows, cols))
                    image = torch.from_numpy(np.copy(image)).float().unsqueeze(0) / 255
                    images.append(image)
        except (FileNotFoundError, ValueError, struct.error):
            print("Error loading images")

        return images

    def load_labels(self):
        labels = []
        try:
            with open(self.label_file, "rb") as file:
                magic, num_labels = struct.unpack(">II", file.read(8))
                label_data = file.read()

                if len(label_data) > 0:
                    labels = np.frombuffer(label_data, dtype=np.uint8)
        except (FileNotFoundError, ValueError, struct.error):
            print("Error loading labels")

        return labels if isinstance(labels, list) else labels.tolist()

    def add_image_label(self, image, label):
        image = torch.squeeze(image, dim=0)
        label = torch.squeeze(label, dim=0)

        self.images.append(image)
        self.labels.append(label)

    def save_to_files(self):
        with open(self.image_file, "wb") as f_image:
            f_image.write(struct.pack(">IIII", 2051, len(self.images), 28, 28))
            for img in self.images:
                img = img.detach().cpu()
                img_array = (img * 254.9).squeeze(0).numpy().astype(np.uint8)
                f_image.write(img_array.tobytes())

        with open(self.label_file, "wb") as f_label:
            f_label.write(struct.pack(">II", 2049, len(self.labels)))
            for lbl in self.labels:
                f_label.write(struct.pack("B", lbl))

    def clear_dataset(self):
        if os.path.exists(self.image_file):
            os.remove(self.image_file)
        if os.path.exists(self.label_file):
            os.remove(self.label_file)

        self.images = []  # Clear the images
        self.labels = []  # Clear the labels
        self.save_to_files()
        print("Dataset cleared")


class DS2(Dataset):
    def __init__(
        self,
        image_file="./data/augmented_train_data/train-images-aug-idx3-ubyte",
        label_file="./data/augmented_train_data/train-labels-aug-idx1-ubyte",
        transform=None,
    ):
        self.image_file = image_file
        self.label_file = label_file
        self.images = self.load_images()
        self.labels = self.load_labels()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

    def load_images(self):
        images = []
        try:
            with open(self.image_file, "rb") as file:
                magic, num_images, rows, cols = struct.unpack(">IIII", file.read(16))
                image_data = file.read()

                for i in range(num_images):
                    image = np.frombuffer(
                        image_data[i * rows * cols : (i + 1) * rows * cols],
                        dtype=np.uint8,
                    )

                    image = np.reshape(image, (rows, cols))
                    image = torch.from_numpy(np.copy(image)).float().unsqueeze(0) / 255
                    images.append(image)
        except (FileNotFoundError, ValueError, struct.error):
            print("Error loading images")

        return images

    def load_labels(self):
        labels = []

        try:
            with open(self.label_file, "rb") as file:
                magic, num_labels = struct.unpack(">II", file.read(8))
                label_data = file.read()

                if len(label_data) > 0:
                    labels = np.frombuffer(label_data, dtype=np.uint8)
        except (FileNotFoundError, ValueError, struct.error):
            print("Error loading labels")

        return labels if isinstance(labels, list) else labels.tolist()

    def add_image_label(self, image, label):
        image = torch.squeeze(image, dim=0)
        label = torch.squeeze(label, dim=0)

        self.images.append(image)
        self.labels.append(label)

    def save_to_files(self):
        with open(self.image_file, "wb") as f_image:
            f_image.write(struct.pack(">IIII", 2051, len(self.images), 28, 28))
            for img in self.images:
                img = img.detach().cpu()
                img_array = (img * 254).squeeze(0).numpy().astype(np.uint8)
                # print(img_array)
                # plt.imshow(img_array)
                # plt.title("save")
                # plt.show()
                f_image.write(img_array.tobytes())

        with open(self.label_file, "wb") as f_label:
            f_label.write(struct.pack(">II", 2049, len(self.labels)))
            for lbl in self.labels:
                f_label.write(struct.pack("B", lbl))

    def clear_dataset(self):
        # Delete the files if they exist
        if os.path.exists(self.image_file):
            os.remove(self.image_file)
        if os.path.exists(self.label_file):
            os.remove(self.label_file)

        # Save empty data to the files after deletion
        self.images = []  # Clear the images
        self.labels = []  # Clear the labels
        self.save_to_files()
        print("Dataset cleared")


def image_to_tensor(image):
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    return image


def recursive_pad(image, rows, cols, border):
    if border <= 0:
        return image
    # Pad top and bottom
    top = np.zeros((1, cols)).reshape(1, 1, cols)
    bottom = np.zeros((1, cols)).reshape(1, 1, cols)
    image = np.concatenate((top, image, bottom), axis=1)

    # Pad left and right
    image = np.transpose(image, (0, 2, 1))
    left = np.zeros((1, rows + 2)).reshape(1, 1, rows + 2)
    right = np.zeros((1, rows + 2)).reshape(1, 1, rows + 2)

    image = np.concatenate((left, image, right), axis=1)
    image = np.transpose(image, (0, 2, 1))

    return recursive_pad(image, rows + 2, cols + 2, border - 1)


def pad_image(image, border):
    """Pad image with zeros"""
    image = image.copy()
    image = image.astype(np.float32)
    rows, cols = image.shape[1:3]

    padded_image = recursive_pad(image, rows, cols, border)

    return padded_image


def scale_down_image(image, size_rows, size_cols):
    image = image.copy()
    image = image.astype(np.float32)
    rows, cols = image.shape[1:3]
    image = image.reshape(rows, cols)
    image = cv2.resize(image, (size_rows, size_cols))
    image = image.reshape(1, size_rows, size_cols)

    return image


def skew(image, factor):
    """Skew image"""
    image = image.copy()
    image_np = image.astype(np.float32)
    rows, cols = image_np.shape[1:3]
    size = (rows + cols) / 2  # size of the image

    pad_border = int(size / 2)
    pad_border = 0
    image_pad = pad_image(
        image, pad_border
    )  # pad and then skew, double size, 4x pixels
    image_pad_np = image_pad.astype(np.float32)
    rows_pad, cols_pad = image_pad_np.shape[1:3]
    size_pad = (rows_pad + cols_pad) / 2  # size of the image

    # random corner offsets
    start = 0
    end = factor * size_pad * 0.6
    c1 = int(np.random.uniform(start, end))
    c2 = int(np.random.uniform(start, end))
    c3 = int(np.random.uniform(start, end))
    c4 = int(np.random.uniform(start, end))

    # Skew image
    pts1 = np.float32([[0, 0], [cols_pad, 0], [0, rows_pad], [cols_pad, rows_pad]])
    pts2 = np.float32(
        [
            [c1, c1],
            [cols_pad - c2, c2],
            [c3, rows_pad - c3],
            [cols_pad - c4, rows_pad - c4],
        ]
    )

    M = cv2.getPerspectiveTransform(pts2, pts1)

    # Apply the perspective transformation to the padded image
    image_reshaped = image_pad_np.reshape(rows_pad, cols_pad)
    image_skewed = cv2.warpPerspective(image_reshaped, M, (cols, rows))

    # Scale down the skewed image to the original dimensions
    image = cv2.resize(image_skewed, (cols, rows), interpolation=cv2.INTER_LINEAR)
    reshaped_output = image.reshape(1, cols, rows)

    return reshaped_output


def rotate(image):
    """Rotate image"""
    image = image.copy()
    image_np = image.astype(np.float32)
    rows, cols = image_np.shape[1:3]
    size = (rows + cols) / 2  # size of the image

    pad_border = int(size / 2)
    pad_border = 0
    image_pad = pad_image(
        image, pad_border
    )  # pad and then skew, double size, 4x pixels
    image_pad_np = image_pad.astype(np.float32)
    rows_pad, cols_pad = image_pad_np.shape[1:3]
    size_pad = (rows_pad + cols_pad) / 2  # size of the image

    angle = 25 * np.random.uniform(-1, 1)
    center = (cols_pad / 2, rows_pad / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    image_reshaped = image_pad_np.reshape(rows_pad, cols_pad)
    image_rotated = cv2.warpAffine(image_reshaped, M, (cols_pad, rows_pad))
    image = cv2.resize(image_rotated, (cols, rows), interpolation=cv2.INTER_LINEAR)
    reshaped_output = image.reshape(1, cols, rows)

    return reshaped_output


def blur(image):
    """Blur image"""
    image = image.copy()
    image_np = image.astype(np.float32)
    rows, cols = image_np.shape[1:3]

    # Blur image
    image_reshaped = image_np.reshape(rows, cols)
    image_blurred = cv2.blur(image_reshaped, (4, 4))
    reshaped_output = image_blurred.reshape(1, cols, rows)

    return reshaped_output


def sharpen(image):
    """Sharpen image"""
    image = image.copy()
    image_np = image.astype(np.float32)
    rows, cols = image_np.shape[1:3]

    # Sharpen image
    image_reshaped = image_np.reshape(rows, cols)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image_sharpened = cv2.filter2D(image_reshaped, -1, kernel)
    reshaped_output = image_sharpened.reshape(1, cols, rows)

    return reshaped_output


def augment_image(image):
    """Augment image"""
    image = image.copy()
    if image.shape != (1, 28, 28):
        image = image.reshape(1, 28, 28)
    image_np = image.astype(np.float32)
    rows, cols = image_np.shape[1:3]

    images = []
    images += [image_np]

    # Pad image
    # image_pad = pad_image(image_np, 10)
    # image_pad = scale_down_image(image_pad, 28, 28)
    # images += [image_pad]

    # Skew image
    image_skewed = skew(image_np, 0.5)
    images += [image_skewed]

    # Rotate image
    image_rotated = rotate(image_np)
    images += [image_rotated]

    # Blur image
    image_blurred = blur(image_np)
    images += [image_blurred]

    # Cross augmentations
    tensor_images = []
    for image in images:
        tensor_images += [image_to_tensor(image)]

        paddded = pad_image(image, 10)
        paddded = scale_down_image(paddded, 28, 28)
        tensor_images += [image_to_tensor(paddded)]

        # skewed = skew(image, 0.5)
        # tensor_images += [image_to_tensor(skewed)]

        rotated = rotate(image)
        tensor_images += [image_to_tensor(rotated)]

        blurred = blur(image)
        tensor_images += [image_to_tensor(blurred)]

    return tensor_images
