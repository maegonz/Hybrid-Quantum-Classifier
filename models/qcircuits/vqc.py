import torch
import torch.nn as nn
import numpy as np
import pennylane as qml


def quantum_net(inputs, weights, n_qubits):
    """
    inputs: tenseur de forme (n_qubits,) -> caractéristiques classiques
    weights: tenseur de forme (n_layers, n_qubits, 3) -> paramètres entraînables
    n_qubits: nombre de qubits
    """
    n_layers = weights.shape[0]

    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)  # Encodage des données classiques dans les qubits
    
    for layer in range(n_layers):
        # Data Re-uploading : On encode les données en les combinant aux poids
        for i in range(n_qubits):
            # Rotation arbitraire combinant l'input classique et les poids quantiques
            qml.Rot(
                inputs[i] * weights[layer, i, 0], 
                weights[layer, i, 1], 
                weights[layer, i, 2], 
                wires=i
            )
        
        # Intrication entre les qubits voisins (génère la non-linéarité quantique)
        if n_qubits > 1:
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0]) # Fermeture de la boucle

    # Mesure sur la base PauliZ pour chaque qubit
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]