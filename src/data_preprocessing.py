"""
Data preprocessing module for Cats vs Dogs classification.
Handles image loading, preprocessing, and data augmentation.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Optional
import shutil


class CatsDogsDataset(Dataset):
    """Custom Dataset for Cats vs Dogs classification."""

    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of labels (0 for cat, 1 for dog)
            transform: Optional transform to be applied on images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transforms():
    """Get training data augmentation transforms."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms():
    """Get validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess a single image for inference.

    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (width, height)

    Returns:
        Preprocessed image as numpy array
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size)
        image_array = np.array(image) / 255.0
        return image_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


def preprocess_image_tensor(image_path: str) -> torch.Tensor:
    """
    Preprocess a single image and return as a tensor for model inference.

    Args:
        image_path: Path to the image file

    Returns:
        Preprocessed image tensor with batch dimension
    """
    transform = get_val_transforms()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Add batch dimension


def create_data_splits(data_dir: str,
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.1,
                       test_ratio: float = 0.1,
                       seed: int = 42) -> Tuple[list, list, list, list, list, list]:
    """
    Create train/val/test splits from a directory structure.

    Expected directory structure:
        data_dir/
            cat/
                image1.jpg
                image2.jpg
                ...
            dog/
                image1.jpg
                image2.jpg
                ...

    Args:
        data_dir: Root directory containing cat and dog subdirectories
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)
    """
    np.random.seed(seed)

    # Collect all image paths and labels
    all_paths = []
    all_labels = []

    class_mapping = {'cat': 0, 'dog': 1}

    for class_name, label in class_mapping.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_paths.append(os.path.join(class_dir, img_name))
                all_labels.append(label)

    # Shuffle data
    indices = np.random.permutation(len(all_paths))
    all_paths = [all_paths[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]

    # Calculate split indices
    n_total = len(all_paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Split data
    train_paths = all_paths[:n_train]
    train_labels = all_labels[:n_train]

    val_paths = all_paths[n_train:n_train + n_val]
    val_labels = all_labels[n_train:n_train + n_val]

    test_paths = all_paths[n_train + n_val:]
    test_labels = all_labels[n_train + n_val:]

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def get_data_loaders(train_paths, train_labels,
                     val_paths, val_labels,
                     batch_size: int = 32,
                     num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        train_paths: List of training image paths
        train_labels: List of training labels
        val_paths: List of validation image paths
        val_labels: List of validation labels
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = CatsDogsDataset(train_paths, train_labels, transform=get_train_transforms())
    val_dataset = CatsDogsDataset(val_paths, val_labels, transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
