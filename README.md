# FashionMNIST CNN Classifier

## Project Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify clothing images from the FashionMNIST dataset.

The objective of the project is to automate product tagging for an e-commerce platform by categorizing fashion items into predefined classes such as shirts, trousers, sneakers, bags, and more.

This project was completed as part of a DataCamp deep learning exercise focused on:

* Computer Vision
* Convolutional Neural Networks (CNNs)
* PyTorch Fundamentals
* Multiclass Classification
* Model Evaluation Metrics

---

# Dataset

The project uses the FashionMNIST dataset.

Dataset characteristics:

* 70,000 grayscale images
* Image size: 28 × 28
* 10 clothing categories
* Training set: 60,000 images
* Test set: 10,000 images

Classes:

| Label | Class       |
| ----- | ----------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

---

# Technologies Used

* Python
* PyTorch
* TorchMetrics
* NumPy

---

# CNN Architecture

The model uses a lightweight CNN architecture for fast training and efficient classification.

Architecture:

```text
Input Image (1×28×28)
↓
Convolution Layer (16 Filters)
↓
ReLU Activation
↓
Max Pooling
↓
Flatten
↓
Fully Connected Layer
↓
10 Output Classes
```

---

# Full Code

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall

# Create dataloaders
train_loader = DataLoader(
    train_data,
    batch_size=256,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=256,
    shuffle=False
)

# Define CNN model
class FashionCNN(nn.Module):

    def __init__(self):
        super(FashionCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                16 * 14 * 14,
                10
            )

        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

# Initialize model
model = FashionCNN()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training loop
epochs = 1

model.train()

for epoch in range(epochs):

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

# Generate predictions
model.eval()

predictions = []
targets = []

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        preds = torch.argmax(outputs, dim=1)

        predictions.extend(preds.tolist())

        targets.extend(labels.tolist())

# Convert to tensors
predictions_tensor = torch.tensor(predictions)

targets_tensor = torch.tensor(targets)

# Metrics
accuracy_metric = Accuracy(
    task="multiclass",
    num_classes=10
)

precision_metric = Precision(
    task="multiclass",
    num_classes=10,
    average=None
)

recall_metric = Recall(
    task="multiclass",
    num_classes=10,
    average=None
)

# Calculate metrics
accuracy = float(
    accuracy_metric(
        predictions_tensor,
        targets_tensor
    )
)

precision = precision_metric(
    predictions_tensor,
    targets_tensor
).tolist()

recall = recall_metric(
    predictions_tensor,
    targets_tensor
).tolist()

print("Accuracy:", accuracy)

print("Precision:", precision)

print("Recall:", recall)
```

---

# Training Process

The model training pipeline includes:

1. Loading FashionMNIST images using DataLoader
2. Passing images through convolution layers
3. Extracting image features using CNN filters
4. Applying ReLU activation
5. Reducing spatial dimensions using max pooling
6. Flattening extracted features
7. Performing multiclass classification
8. Updating model parameters using backpropagation and Adam optimizer

---

# Evaluation Metrics

The project evaluates model performance using:

## Accuracy

Measures the percentage of correctly classified images.

Formula:

```text
Accuracy = Correct Predictions / Total Predictions
```

---

## Precision

Measures how many predicted labels are actually correct.

Formula:

```text
Precision = TP / (TP + FP)
```

Where:

* TP = True Positives
* FP = False Positives

---

## Recall

Measures how many actual samples were correctly detected.

Formula:

```text
Recall = TP / (TP + FN)
```

Where:

* FN = False Negatives

---

# Key Deep Learning Concepts Used

* Convolutional Neural Networks (CNNs)
* Feature Extraction
* ReLU Activation Function
* Max Pooling
* Forward Propagation
* Backpropagation
* Gradient Descent Optimization
* Multiclass Classification
* Model Evaluation

---

# Results

The CNN model successfully:

* Learned image features from fashion products
* Generated predictions for unseen test images
* Calculated accuracy, precision, and recall metrics
* Performed automated image classification efficiently

---

# Learning Outcomes

Through this project, the following skills were developed:

* Building CNNs using PyTorch
* Working with image datasets
* Implementing training loops
* Using DataLoaders efficiently
* Evaluating classification models
* Understanding CNN architecture fundamentals
* Applying deep learning to computer vision tasks

---

# Future Improvements

Potential improvements for this project include:

* Adding additional convolution layers
* Using Batch Normalization
* Applying Dropout regularization
* Training for more epochs
* Using GPU acceleration
* Implementing data augmentation
* Experimenting with deeper CNN architectures

