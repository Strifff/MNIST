import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from utils.data_utils import image_to_tensor


def plot_grid(data):
    
    size = len(data)
    grid_size = 1
    while grid_size * grid_size < size:
        grid_size += 1
    
    grid_size_x = grid_size
    grid_size_y = 1
    while grid_size_x * grid_size_y < size:
        grid_size_y += 1
    
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(grid_size_y, grid_size_x, figsize=(8, 8))
    
    print("grid_size_x", grid_size_x)
    print("grid_size_y", grid_size_y)

    for row in range(grid_size_y):
        for col in range(grid_size_x):
            index = row * grid_size_x + col
            i = row
            j = col
            print("row: ", row, "col: ", col, "index: ", index)
            if index >= len(data):
                break
            image, label = data[index]
            #print(image.shape)

            sns.heatmap(image[0][0], cmap="gray", ax=axes[i, j], cbar=False)
            axes[i, j].set_title(f"Label: {label}")
            axes[i, j].axis("off")

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()
