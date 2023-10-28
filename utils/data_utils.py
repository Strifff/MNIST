import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt


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

    print(start, end)

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

    angle = 45 * np.random.uniform(-1, 1)
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
