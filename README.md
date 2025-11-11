# Fashion-MNIST Neural Network Classifier

A PyTorch implementation of neural network models for classifying Fashion-MNIST dataset images. This project includes both a basic and a deeper neural network architecture for image classification.

## Overview

This project implements neural networks to classify 28x28 grayscale images from the Fashion-MNIST dataset into 10 categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot

## Requirements

- Python 3.8 or higher
- PyTorch 2.1.0
- torchvision 0.16.0
- torchaudio 2.1.0

## Installation

### Setting Up Python Virtual Environment

#### Step 1: Create Virtual Environment

```bash
python3 -m venv Pytorch
```

#### Step 2: Activate Virtual Environment

**On Linux/Mac:**
```bash
source Pytorch/bin/activate
```

**On Windows:**
```bash
Pytorch\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

#### Verify Installation

```bash
pip freeze
```

You should see the installed packages listed.

### Deactivating Virtual Environment

When you're done working:
```bash
deactivate
```

## Project Structure

```
.
├── intro_pytorch.py       # Main implementation file
├── data/                  # Dataset directory (created automatically)
└── README.md             # This file
```

## Usage

### Importing the Module

```python
from intro_pytorch import (
    get_data_loader,
    build_model,
    build_deeper_model,
    train_model,
    evaluate_model,
    predict_label
)
import torch.nn as nn
```

### Basic Workflow

#### 1. Load Data
```python
# Get training data loader
train_loader = get_data_loader(training=True)

# Get test data loader
test_loader = get_data_loader(training=False)
```

#### 2. Build Model
```python
# Basic model (128 -> 64 -> 10 nodes)
model = build_model()

# Or deeper model (256 -> 128 -> 64 -> 32 -> 10 nodes)
model = build_deeper_model()
```

#### 3. Train Model
```python
criterion = nn.CrossEntropyLoss()
train_model(model, train_loader, criterion, T=5)  # Train for 5 epochs
```

Expected output format:
```
Train Epoch: 0 Accuracy: 42954/60000(71.59%) Loss: 0.833
Train Epoch: 1 Accuracy: 49602/60000(82.67%) Loss: 0.489
...
```

#### 4. Evaluate Model
```python
# Show both loss and accuracy
evaluate_model(model, test_loader, criterion, show_loss=True)

# Show only accuracy
evaluate_model(model, test_loader, criterion, show_loss=False)
```

Expected output:
```
Average loss: 0.4116
Accuracy: 85.39%
```

#### 5. Make Predictions
```python
# Get a batch of test images
test_images = next(iter(test_loader))[0]

# Predict label for image at index 1
predict_label(model, test_images, 1)
```

Expected output:
```
Pullover: 92.48%
Shirt: 5.93%
Coat: 1.48%
```

## API Reference

### `get_data_loader(training=True)`
Returns a DataLoader for the Fashion-MNIST dataset.
- **Parameters:** `training` (bool) - If True, returns training set loader; if False, returns test set loader
- **Returns:** torch.utils.data.DataLoader object with batch_size=64

### `build_model()`
Constructs a basic neural network with 3 dense layers.
- **Returns:** Untrained PyTorch Sequential model

### `build_deeper_model()`
Constructs a deeper neural network with 5 dense layers.
- **Returns:** Untrained PyTorch Sequential model

### `train_model(model, train_loader, criterion, T)`
Trains the model for T epochs using SGD optimizer.
- **Parameters:**
  - `model`: Neural network model
  - `train_loader`: Training data loader
  - `criterion`: Loss function (e.g., nn.CrossEntropyLoss())
  - `T`: Number of training epochs
- **Returns:** None (prints training progress)

### `evaluate_model(model, test_loader, criterion, show_loss=True)`
Evaluates the trained model on test data.
- **Parameters:**
  - `model`: Trained neural network model
  - `test_loader`: Test data loader
  - `criterion`: Loss function
  - `show_loss`: If True, displays loss; if False, shows only accuracy
- **Returns:** None (prints evaluation metrics)

### `predict_label(model, test_images, index)`
Predicts the top 3 most likely labels for a specific image.
- **Parameters:**
  - `model`: Trained neural network model
  - `test_images`: Tensor of shape Nx1x28x28
  - `index`: Index of the image to predict
- **Returns:** None (prints top 3 predictions with probabilities)

## Model Architecture

### Basic Model
```
Input (28x28) → Flatten → Dense(128) → ReLU → Dense(64) → ReLU → Dense(10)
```

### Deeper Model
```
Input (28x28) → Flatten → Dense(256) → ReLU → Dense(128) → ReLU → Dense(64) → ReLU → Dense(32) → ReLU → Dense(10)
```

## Training Details

- **Optimizer:** SGD with learning rate 0.001 and momentum 0.9
- **Loss Function:** CrossEntropyLoss
- **Batch Size:** 64
- **Expected Accuracy:** >80% after 5 epochs (basic model), >85% achievable with tuning

## Dataset

The Fashion-MNIST dataset is automatically downloaded on first run and stored in the `./data` directory. It contains:
- **Training set:** 60,000 images
- **Test set:** 10,000 images
- **Image size:** 28x28 grayscale
- **Classes:** 10 fashion categories

## Troubleshooting

### Import Errors
Ensure you're using only torch, torchvision, and Python standard library imports. No additional packages are allowed.

### Dataset Download Issues
If the dataset fails to download automatically, check your internet connection and ensure the `./data` directory has write permissions.

### Version Compatibility
This project is tested with PyTorch 2.1.0. Using different versions may cause compatibility issues.

## License

MIT

## Acknowledgments

- Fashion-MNIST dataset by Zalando Research
- PyTorch framework by Meta AI
