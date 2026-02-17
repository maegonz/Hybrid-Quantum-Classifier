# Hybrid-Quantum-Classification

## Overview

This repository explores **hybrid quantum–classical classification models**, combining the strengths of **classical deep learning** with **quantum machine learning**.  
The goal is to implement and evaluate different **classification tasks** using hybrid architectures where quantum circuits are embedded as trainable layers inside classical neural networks.

Hybrid models are particularly useful in researching:
- Near-term quantum algorithms (NISQ era)
- Quantum feature extraction
- Variational Quantum Circuits (VQCs) integrated with classical optimization

This repository serves as both a **research playground** and a **learning resource** for hybrid quantum machine learning.


## Features

- ✅ Hybrid quantum–classical models using **PyTorch + PennyLane**
- ✅ Multiple classification tasks (binary and multi-class)
- ✅ Variational Quantum Circuits (VQCs)
- ✅ End-to-end training with classical optimizers
- ✅ Modular and extensible codebase
- ✅ CPU-based quantum simulators (no quantum hardware required)


## Structure

```text
Hybrid-Quantum-Classification/
│
├── data/
│   ├── datasets.py          # Dataset loading and preprocessing
│
├── models/
│   ├── classical.py         # Pure classical baselines
│   ├── quantum.py           # Quantum circuit definitions
│   ├── hybrid.py            # Hybrid quantum-classical models
│
├── training/
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Evaluation metrics
│
├── experiments/
│   ├── binary_classification.ipynb
│   ├── multiclass_classification.ipynb
│
├── utils/
│   ├── config.py            # Hyperparameters and configs
│   ├── metrics.py           # Accuracy, loss, etc.
│
├── requirements.txt
├── README.md
└── main.py
```

## Hybrid Model Architecture

A typical hybrid model in this repository follows this pipeline:

Classical preprocessing layer (PyTorch)

Quantum layer

Data encoding into quantum states

Variational quantum circuit (trainable parameters)

Classical post-processing layer

Loss computation & optimization using classical optimizers

```
Input → Classical Layer → Quantum Circuit → Classical Layer → Output
```

## Classification Tasks

Implemented tasks include:

🔹 Binary classification (e.g., toy datasets, synthetic data)

🔹 Multi-class classification

🔹 Quantum-enhanced feature learning

🔹 Comparison with purely classical baselines

Each task is designed to highlight how quantum layers influence performance and learning behavior.

## Installation
Install Dependencies
```
pip install -r requirements.txt
```

## Results

Training curves and accuracy metrics are logged
![](/imgs/output_0.png)
<!-- ![](/imgs/output.png) -->


Jupyter notebooks in experiments/ provide visual analysis

Classical vs hybrid performance comparisons included

## Future Work

🔹 More expressive quantum ansätze

🔹 Larger datasets and feature maps

🔹 Quantum convolutional layers

🔹 Benchmarking against state-of-the-art classical models

<!-- ## Author

Author:  -->

## License

This project is licensed under the MIT License.
Feel free to use, modify, and distribute for research and educational purposes.