import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import Optimizer
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from IPython.display import clear_output
from utils import plot_decision_boundary, plot_training_curves


def training(model: nn.Module,
             X,
             y,
             criterion: nn.Module,
             optimizer: Optimizer,
             device: torch.device,
             epochs: int,
             use_amp: bool=True,
             visualize: bool=False,
             refresh_rate: int=5,
             save_path: str='./models/progress_steps'):
    """
    Train a PyTorch model with optional Automatic Mixed Precision.

    Parameters
    ----------
    model : nn.Module
        The neural network model to be trained.
    X : torch.Tensor
        Input features for training.
    y : torch.Tensor
        Target labels for training.
    criterion : nn.Module
        Loss function used to compute training loss.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    device : torch.device
        Device on which to train the model ('cpu' or 'cuda').
    epochs : int
        Number of training epochs.
    visualize : bool, optional
        Whether to visualize the training process. Default is False.
    use_amp : bool, optional
        Whether to use AMP.
        AMP is enabled only when using a CUDA device. Default is True.

    Returns
    -------
    train_losses : list of float
        Average training loss for each epoch.
    train_accuracies : list of float
        Training accuracy (percentage) for each epoch.
    """

    model.to(device)
    X, y = X.to(device), y.to(device)

    use_amp = use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    loss_history, acc_history = [], []

    if visualize:
        # Préparation de la grille pour la frontière de décision (calculée une seule fois)
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

    epoch_tqdm = tqdm(range(epochs), desc="Training Progress")

    for epoch in epoch_tqdm:
        model.train()

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(X)
            y = y.view_as(outputs)  # Ensure y has the same shape as outputs
            loss = criterion(outputs, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            pred_labels = (outputs > 0.5).float()
            acc = accuracy_score(y.numpy(), pred_labels.numpy())
            
            loss_history.append(loss.item())
            acc_history.append(acc)

        epoch_loss = sum(loss_history) / len(loss_history)
        epoch_acc = sum(acc_history) / len(acc_history)
        epoch_tqdm.set_postfix(train_loss=epoch_loss, train_acc=epoch_acc)

        if visualize and ((epoch + 1) % refresh_rate == 0 or epoch == epochs - 1):
            # clear_output(wait=True)  # Clear and update the plot
            plot_decision_boundary(model, X, y, grid_tensor, xx, yy, save_path=f'{save_path}/decision_boundary_epoch_{epoch+1}.png')
            plot_training_curves(loss_history, acc_history, save_path=f'{save_path}/training_curves_epoch_{epoch+1}.png')

    return loss_history, acc_history

        
def evaluating(model: nn.Module, 
               data_loader: DataLoader,
               criterion: nn.Module,
               device: torch.device,
               use_amp: bool = True):
    """
    Evaluate a PyTorch model on a dataset with optional AMP.

    Parameters
    ----------
    model : nn.Module
        The trained model to be evaluated.
    data_loader : DataLoader
        DataLoader providing the evaluation dataset.
    criterion : nn.Module
        Loss function used to compute evaluation loss.
    device : torch.device
        Device on which evaluation is performed ('cpu' or 'cuda').
    use_amp : bool, optional
        Whether to use Automatic Mixed Precision (AMP).
        AMP is enabled only when using a CUDA device. Default is True.

    Returns
    -------
    avg_loss : float
        Average loss over the entire dataset.
    avg_accuracy : float
        Average accuracy (percentage) over the entire dataset.
    """

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            predicted = torch.argmax(outputs, dim=1)
            accuracy += (predicted == labels).sum().item()

            del outputs, loss, imgs, labels

        torch.cuda.empty_cache()

    avg_loss = total_loss / len(data_loader.dataset)
    avg_accuracy = 100.0 * accuracy / len(data_loader.dataset)
    return avg_loss, avg_accuracy


# def prediction(model: nn.Module, 
#                data_loader: DataLoader,
#                device: torch.device,
#                cm: bool = True):
#     """
#     Evaluate a PyTorch model on a dataset with optional AMP.

#     Parameters
#     ----------
#     model : nn.Module
#         The trained model to be evaluated.
#     data_loader : DataLoader
#         DataLoader providing the evaluation dataset.
#     device : torch.device
#         Device on which evaluation is performed ('cpu' or 'cuda').
#     cm : bool, optional
#         Whether to display the confusion matrix.
#         Default is True.

#     Returns
#     -------
#     all_preds : list[float]
#         All predicted masks over the entire dataset.
#     all_masks : list[float]
#         All true masks over the entire dataset.
#     """
#     model.eval()
#     all_preds = []
#     all_masks = []
#     with torch.no_grad():
#         for imgs, masks in data_loader:
#             imgs, masks = imgs.to(device), masks.to(device)
#             outputs = model(imgs)
#             _, predicted = torch.max(outputs, 1)
#             all_preds.extend(predicted.cpu().numpy())
#             all_masks.extend(masks.cpu().numpy())
    
#     if cm:
#         cm = confusion_matrix(all_masks, all_preds)
#         sns.heatmap(cm, fmt='d', cmap='YlGnBu')
#         plt.xlabel('Predicted')
#         plt.ylabel('True Label')
#         plt.title('Confusion Matrix')
#         plt.show()
    
#     return all_preds, all_masks