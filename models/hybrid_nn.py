import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from .qcircuits.qlayer import QuantumLayer

class HybridResNet(nn.Module):
    def __init__(self, n_qubits=4, qcircuit=None, num_classes=2):
        super().__init__()
        # Load pre-trained ResNet18
        self.resnet = models.resnet18(pretrained=True)
        
        # Freeze ResNet parameters (We don't want to retrain the vision part)
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Map ResNet output into 4 qubits
        self.fc = nn.Linear(512, n_qubits)
        
        # Quantum Layer
        self.qlayer = QuantumLayer(n_qubits=n_qubits, n_layers=3, qcircuit=qcircuit)
        
        # Classification layer
        self.fc_final = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        x = self.resnet(x)       # Classical feature extraction
        x = self.fc(x)
        x = torch.tanh(x)        # Normalization to [-1, 1] for embedding
        x = self.qlayer(x)
        x = self.fc_final(x)
        return torch.softmax(x, dim=1)

class HybridBinaryModel(nn.Module):
    def __init__(self, n_qubits=4, n_layers=3, qcircuit=None, num_classes=1):
        super().__init__()
        # Quantum Layer       
        self.qlayer = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers, qcircuit=qcircuit)

        # Classification layer (for binary classification, we can use a single output with sigmoid)
        self.fc_final = nn.Sequential(
            nn.Linear(n_qubits, 32),  
            nn.ReLU(),
            nn.Linear(32, num_classes)  # Output for multi-class classification
        )
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        x = self.qlayer(x)
        x = x.to(torch.float32)  # Ensure the output is in float32 for the final layer
        # print("Output from Quantum Layer:", x.shape)
        x = self.fc_final(x)
        # return torch.sigmoid(x)  # Sigmoid activation for binary classification
        return x  # Return raw logits for BCEWithLogitsLoss