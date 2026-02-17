import torch
import torch.nn as nn
import pennylane as qml


class QuantumLayer(nn.Module):
    def __init__(self, num_qubits, num_qlayers, qcircuit=None):
        super().__init__()
        self.num_qubits = num_qubits
        self.qdevice = qml.device("default.qubit", wires=num_qubits)
        
        weight_shapes = {"weights": (num_qlayers, num_qubits, 3)}
        
        if qcircuit is not None:
            self.qcircuit = qml.QNode(qcircuit, self.qdevice, interface="torch")
        else:
            self.qcircuit = qml.QNode(self._circuit, self.qdevice, interface="torch")
        
        self.torch_layer = qml.qnn.TorchLayer(self.qcircuit, weight_shapes)

    def _circuit(self, inputs, weights):
        """internal method to define the quantum circuit"""
        # Encode classical data into quantum states (Angle Embedding is standard)
        qml.AngleEmbedding(inputs, wires=range(self.num_qubits))
        # Variational layers
        qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def forward(self, x):
        return self.torch_layer(x)