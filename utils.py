import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from IPython.display import clear_output  # Pour effacer et mettre à jour le graphique

def plot_imgs(X, y, n=10):
    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.show()

# def plot_decision_boundary(model, X, y):
#     # Calcul de la frontière pour l'état actuel du modèle
#     with torch.no_grad():
#         Z = model(grid_tensor).reshape(xx.shape)
    
#     # --- CRÉATION DE LA FIGURE ---
#     clear_output(wait=True) # Efface la sortie précédente
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
#     # GAUCHE : Frontière de Décision
#     ax1.set_title(f"Decision Boundary (Epoch {epoch+1})")
#     ax1.contourf(xx, yy, Z, levels=20, cmap="RdBu", alpha=0.8)
#     ax1.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap="RdBu_r", edgecolors='k')
    
#     # DROITE : Courbes Loss & Accuracy
#     ax2.set_title("Training Metrics")
#     ax2.plot(loss_history, label="Loss", color="red", linewidth=2)
#     ax2.plot(acc_history, label="Accuracy", color="blue", linewidth=2)
#     ax2.set_xlabel("Epochs")
#     ax2.set_ylim(0, 1.1) # Car l'accuracy est entre 0 et 1
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     plt.show()


#     # Create a grid of points to evaluate the model
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
#                           np.arange(y_min, y_max, 0.01))
#     grid_points = np.c_[xx.ravel(), yy.ravel()]
#     grid_tensor = torch.from_numpy(grid_points).float()
#     with torch.no_grad():
#         Z = model(grid_tensor)
#     Z = Z.detach().numpy().reshape(xx.shape)
#     plt.contourf(xx, yy, Z, levels=20, cmap=plt.cm.RdBu, alpha=0.8)
#     plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1)
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu_r, edgecolors='k')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.title('Decision Boundary')
#     plt.show()


def train_and_visualize(model, optimizer, loss_fn, X, y, epochs=50, refresh_rate=5):
    """
    Entraîne le modèle et affiche une animation en direct de la frontière de décision
    et des courbes d'apprentissage.
    
    Args:
        refresh_rate (int): Met à jour le graphique tous les N epochs (pour éviter le clignotement).
    """
    
    # Listes pour stocker l'historique
    loss_history = []
    acc_history = []
    
    # Préparation de la grille pour la frontière de décision (calculée une seule fois)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

    model.train() # Mode entraînement
    
    for epoch in range(epochs):
        # 1. Forward Pass
        optimizer.zero_grad()
        predictions = model(X)
        
        # Correction de forme si nécessaire (ex: [N, 1] vs [N])
        y = y.view_as(predictions) 
        
        loss = loss_fn(predictions, y)
        
        # 2. Backward Pass
        loss.backward()
        optimizer.step()
        
        # 3. Calcul des métriques
        with torch.no_grad():
            pred_labels = (predictions > 0.5).float()
            acc = accuracy_score(y.numpy(), pred_labels.numpy())
            
            loss_history.append(loss.item())
            acc_history.append(acc)

        # 4. Visualisation (Seulement tous les 'refresh_rate' epochs)
        if (epoch + 1) % refresh_rate == 0 or epoch == epochs - 1:
            
            # Calcul de la frontière pour l'état actuel du modèle
            with torch.no_grad():
                Z = model(grid_tensor).reshape(xx.shape)
            
            # --- CRÉATION DE LA FIGURE ---
            clear_output(wait=True) # Efface la sortie précédente
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # GAUCHE : Frontière de Décision
            ax1.set_title(f"Decision Boundary (Epoch {epoch+1})")
            ax1.contourf(xx, yy, Z, levels=20, cmap="RdBu", alpha=0.8)
            ax1.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap="RdBu_r", edgecolors='k')
            
            # DROITE : Courbes Loss & Accuracy
            ax2.set_title("Training Metrics")
            ax2.plot(loss_history, label="Loss", color="red", linewidth=2)
            ax2.plot(acc_history, label="Accuracy", color="blue", linewidth=2)
            ax2.set_xlabel("Epochs")
            ax2.set_ylim(0, 1.1) # Car l'accuracy est entre 0 et 1
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.show()
            
    print("Training Completed!")
    return loss_history, acc_history