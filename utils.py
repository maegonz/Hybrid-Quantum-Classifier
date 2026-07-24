import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from IPython.display import clear_output  # Pour effacer et mettre à jour le graphique


def plot_decision_boundary(model, X, y, grid_points=None, xx=None, yy=None, save_path='./figures/progress_steps/decision_boundary.png'):
    if grid_points is None:
        # Create a grid of points to evaluate the model
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

    with torch.no_grad():
        Z = model(grid_points)
        Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.axis('off')
    plt.title('Decision Boundary')
    plt.savefig(save_path)  # Save the figure
    plt.close()

def plot_training_curves(loss_history, acc_history, save_path='./figures/progress_steps/training_curves.png'):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(loss_history, color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(acc_history, color=color, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.savefig(save_path)
    plt.close()

import glob
from PIL import Image

def create_gif(image_dir: str, output_path: str, duration_ms: int=200, image_pattern: str='decision_boundary'):
    """
    Create a GIF from images in a directory.
    
    Parameters
    ----------
    image_dir : str
        Directory containing the images to be included in the GIF.
    output_path : str
        Path where the GIF will be saved.
    duration_ms : int, optional
        Duration of each frame in milliseconds. Default is 200 ms.
    """
    image_paths = [
        f'{image_dir}/{image_pattern}_epoch_{i}.png' for i in range(1, 351)
        if glob.glob(f'{image_dir}/{image_pattern}_epoch_{i}.png')
    ]
    # print(image_paths)
    print(f"Creating GIF from {len(image_paths)} images in {image_dir}...")
    print(f"Saving GIF to {output_path} with frame duration {duration_ms} ms.")
    if not image_paths:
        raise ValueError(f"No images found in directory: {image_dir}")
    
    images = [Image.open(img) for img in image_paths]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0
    )