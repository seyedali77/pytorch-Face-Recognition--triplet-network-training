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
```

## Usage
Prepare the Dataset:

The project expects the dataset to be organized as follows:
```/path/to/att_faces/
    ├── train/   # Training images organized by class
    └── valid/   # Validation images organized by class
```

**This script will:**

Load images and apply the defined data augmentations.
Create triplets (anchor, positive, negative) for training.
Train the network using the custom triplet loss.
Validate the model on the validation dataset.
Print real-time loss metrics during training and validation.

## Visualize the Data:

The code includes a helper function imshow that displays a grid of images (anchor, positive, and negative). This is useful for ensuring that the data augmentation and triplet creation are working as expected.

## Model Architecture



The triplet network is built on a modified ResNet50:
- **Backbone: A pre-trained ResNet50 model (from torchvision.models) with its final classification layer replaced.**
- **Embedding Head:**
    - A Linear layer reducing the features to 512 dimensions.
    - Batch Normalization.
    - ReLU activation.
    - A final Linear layer projecting to an embedding size (default 128).
   - **Training & Validation**
    - Loss Function: The custom TripletLoss computes the difference between the squared Euclidean distances of the anchor-positive and anchor-negative pairs, enforcing a           margin.
    - Optimizer & Scheduler: The model is optimized using Adam (learning rate 0.0002) with a learning rate scheduler that steps down every 5 epochs.
    - Epochs: Default training runs for 10 epochs, which can be adjusted as needed.
## Contributing



Contributions are welcome! If you have improvements or bug fixes, please:
- Fork the repository.
- Create a new branch for your feature or bug fix.
- Submit a pull request with detailed explanations of your changes.
- For major changes, open an issue first to discuss your approach.
