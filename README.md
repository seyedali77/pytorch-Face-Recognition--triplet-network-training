# pytorch-Face-Recognition--triplet-network-training
This repository contains an end-to-end implementation for training a triplet network using PyTorch. The network learns face embeddings by enforcing that images of the same person are closer in the embedding space than those of different people. 
# PyTorch Triplet Network Training

This repository leverages a pre-trained ResNet50 backbone and a custom triplet loss function to achieve robust feature extraction and discrimination.

## Features

- **Triplet Network Architecture:** Processes anchor, positive, and negative images to learn a meaningful embedding space.
- **Custom Triplet Loss:** Implements a loss function that minimizes the distance between anchor-positive pairs while maximizing the distance to negative examples.
- **Pre-trained Backbone:** Utilizes ResNet50 from `torchvision` with a modified fully-connected layer to produce embeddings.
- **Data Augmentation:** Applies transformations such as resizing, random cropping, horizontal flipping, and rotation to enhance generalization.
- **Training & Validation Pipelines:** Provides detailed routines for both training and validation with real-time progress reporting.

## Requirements

- Python 3.7+
- PyTorch (>=1.8.0)
- torchvision
- matplotlib
- Pillow
- numpy

Install the dependencies using:

```bash
pip install torch torchvision matplotlib pillow numpy


.
├── data_utils.py       # (Optional) Utility functions for data handling
├── train.py            # Main training script containing model, loss, and training loop
├── README.md           # Project documentation
└── [dataset]           # Your dataset directory (e.g., att_faces)

