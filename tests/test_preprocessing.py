"""
Unit tests for data preprocessing functions.
"""

import os
import pytest
import numpy as np
import torch
from PIL import Image
import tempfile
import shutil

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import (
    preprocess_image,
    preprocess_image_tensor,
    get_train_transforms,
    get_val_transforms,
    create_data_splits,
    CatsDogsDataset
)


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    image = Image.new('RGB', (300, 300), color='red')
    return image


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with sample images."""
    temp_dir = tempfile.mkdtemp()

    # Create class directories
    cat_dir = os.path.join(temp_dir, 'cat')
    dog_dir = os.path.join(temp_dir, 'dog')
    os.makedirs(cat_dir)
    os.makedirs(dog_dir)

    # Create sample images
    for i in range(10):
        img = Image.new('RGB', (224, 224), color='red' if i % 2 == 0 else 'blue')
        img.save(os.path.join(cat_dir, f'cat_{i}.jpg'))

    for i in range(10):
        img = Image.new('RGB', (224, 224), color='green' if i % 2 == 0 else 'yellow')
        img.save(os.path.join(dog_dir, f'dog_{i}.jpg'))

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_image_file(sample_image):
    """Create a temporary image file."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    sample_image.save(temp_file.name)
    temp_file.close()

    yield temp_file.name

    # Cleanup
    os.unlink(temp_file.name)


class TestPreprocessing:
    """Test data preprocessing functions."""

    def test_preprocess_image(self, temp_image_file):
        """Test single image preprocessing."""
        result = preprocess_image(temp_image_file, target_size=(224, 224))

        assert isinstance(result, np.ndarray)
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.float64
        assert 0 <= result.min() <= result.max() <= 1

    def test_preprocess_image_different_sizes(self, temp_image_file):
        """Test preprocessing with different target sizes."""
        result = preprocess_image(temp_image_file, target_size=(128, 128))
        assert result.shape == (128, 128, 3)

        result = preprocess_image(temp_image_file, target_size=(256, 256))
        assert result.shape == (256, 256, 3)

    def test_preprocess_image_invalid_path(self):
        """Test preprocessing with invalid image path."""
        with pytest.raises(ValueError):
            preprocess_image('nonexistent_image.jpg')

    def test_preprocess_image_tensor(self, temp_image_file):
        """Test tensor preprocessing for model inference."""
        result = preprocess_image_tensor(temp_image_file)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224
        assert result.dtype == torch.float32

    def test_get_train_transforms(self):
        """Test training transforms."""
        transform = get_train_transforms()
        assert transform is not None

        # Apply to sample image
        img = Image.new('RGB', (300, 300), color='red')
        result = transform(img)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_get_val_transforms(self):
        """Test validation transforms."""
        transform = get_val_transforms()
        assert transform is not None

        # Apply to sample image
        img = Image.new('RGB', (300, 300), color='red')
        result = transform(img)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_create_data_splits(self, temp_data_dir):
        """Test data splitting function."""
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = create_data_splits(
            temp_data_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42
        )

        # Check split sizes
        total = len(train_paths) + len(val_paths) + len(test_paths)
        assert total == 20  # 10 cats + 10 dogs

        assert len(train_paths) == len(train_labels)
        assert len(val_paths) == len(val_labels)
        assert len(test_paths) == len(test_labels)

        # Check labels are valid
        assert all(label in [0, 1] for label in train_labels)
        assert all(label in [0, 1] for label in val_labels)
        assert all(label in [0, 1] for label in test_labels)

        # Check paths exist
        assert all(os.path.exists(path) for path in train_paths)
        assert all(os.path.exists(path) for path in val_paths)
        assert all(os.path.exists(path) for path in test_paths)

    def test_create_data_splits_ratios(self, temp_data_dir):
        """Test data splits with different ratios."""
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = create_data_splits(
            temp_data_dir,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )

        total = len(train_paths) + len(val_paths) + len(test_paths)
        assert total == 20

        # Approximate ratio check (due to rounding)
        assert abs(len(train_paths) / total - 0.7) < 0.1
        assert abs(len(val_paths) / total - 0.15) < 0.1

    def test_dataset_class(self, temp_data_dir):
        """Test CatsDogsDataset class."""
        train_paths, train_labels, _, _, _, _ = create_data_splits(
            temp_data_dir, seed=42
        )

        transform = get_val_transforms()
        dataset = CatsDogsDataset(train_paths, train_labels, transform=transform)

        # Check dataset length
        assert len(dataset) == len(train_paths)

        # Check getitem
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert label in [0, 1]

    def test_dataset_without_transform(self, temp_data_dir):
        """Test dataset without transforms."""
        train_paths, train_labels, _, _, _, _ = create_data_splits(
            temp_data_dir, seed=42
        )

        dataset = CatsDogsDataset(train_paths, train_labels, transform=None)
        image, label = dataset[0]

        assert isinstance(image, Image.Image)
        assert label in [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
