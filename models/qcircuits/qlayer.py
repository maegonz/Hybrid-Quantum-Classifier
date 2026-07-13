import torch
import torch.nn as nn
import pennylane as qml


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers, qcircuit=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.qdevice = qml.device("default.qubit", wires=n_qubits)
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        # Poids quantiques à optimiser par rétropropagation
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3, dtype=torch.float32) * 0.1
        )
        
        if qcircuit is not None:
            self.qcircuit = qml.QNode(qcircuit, self.qdevice, interface="torch")
        else:
            self.qcircuit = qml.QNode(self._circuit, self.qdevice, interface="torch")
        
        # self.torch_layer = qml.qnn.TorchLayer(self.qcircuit, weight_shapes)

    def _circuit(self, inputs, weights, n_qubits):
        """internal method to define the quantum circuit"""
        # Encode classical data into quantum states (Angle Embedding is standard)
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        # Variational layers
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    def forward(self, x):
        # x a une forme (batch_size, n_features). On s'assure que n_features == n_qubits.
        # (Si ce n'est pas le cas, ajoutez un nn.Linear(input_dim, n_qubits) au début).
        
        quantum_outputs = []
        
        # Traitement batch par batch pour le QNode
        for sample in x:
            q_out = self.qcircuit(sample, self.weights, self.n_qubits)
            quantum_outputs.append(torch.stack(q_out))
            
        # Regroupement des résultats sous forme de tenseur (batch_size, n_qubits)
        x_quantized = torch.stack(quantum_outputs)        
        return x_quantized