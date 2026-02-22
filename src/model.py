"""
Model architecture for Cats vs Dogs binary classification.
Implements a baseline CNN model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class BaselineCNN(nn.Module):
    """
    Baseline CNN model for binary image classification.
    Simple architecture with 3 convolutional blocks and 2 fully connected layers.
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        """
        Args:
            num_classes: Number of output classes (default: 2 for binary classification)
            dropout_rate: Dropout rate for regularization
        """
        super(BaselineCNN, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Convolutional Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        # After 4 pooling layers: 224 -> 112 -> 56 -> 28 -> 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """Forward pass through the network."""
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Block 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class SimpleCNN(nn.Module):
    """
    Simplified CNN model for quick baseline.
    """

    def __init__(self, num_classes: int = 2):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(model_name: str = "baseline", **kwargs) -> nn.Module:
    """
    Factory function to get model by name.

    Args:
        model_name: Name of the model ("baseline" or "simple")
        **kwargs: Additional arguments to pass to the model constructor

    Returns:
        PyTorch model instance
    """
    models = {
        "baseline": BaselineCNN,
        "simple": SimpleCNN
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Get a summary of the model architecture.

    Args:
        model: PyTorch model

    Returns:
        Dictionary containing model summary information
    """
    total_params = count_parameters(model)

    return {
        "model_class": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": total_params,
        "model_size_mb": total_params * 4 / (1024 ** 2)  # Assuming float32
    }
