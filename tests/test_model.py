"""
Unit tests for model and inference functions.
"""

import os
import pytest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import tempfile

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import (
    BaselineCNN,
    SimpleCNN,
    get_model,
    count_parameters,
    get_model_summary
)


class TestModelArchitecture:
    """Test model architecture and utilities."""

    def test_baseline_cnn_initialization(self):
        """Test BaselineCNN model initialization."""
        model = BaselineCNN(num_classes=2, dropout_rate=0.5)
        assert isinstance(model, nn.Module)
        assert model is not None

    def test_baseline_cnn_forward_pass(self):
        """Test forward pass through BaselineCNN."""
        model = BaselineCNN(num_classes=2, dropout_rate=0.5)
        model.eval()

        # Create dummy input (batch_size=4, channels=3, height=224, width=224)
        dummy_input = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            output = model(dummy_input)

        # Check output shape
        assert output.shape == (4, 2)  # batch_size=4, num_classes=2

    def test_baseline_cnn_output_range(self):
        """Test that model outputs logits (not probabilities)."""
        model = BaselineCNN(num_classes=2)
        model.eval()

        dummy_input = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output = model(dummy_input)

        # Logits can be any real number
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_simple_cnn_initialization(self):
        """Test SimpleCNN model initialization."""
        model = SimpleCNN(num_classes=2)
        assert isinstance(model, nn.Module)

    def test_simple_cnn_forward_pass(self):
        """Test forward pass through SimpleCNN."""
        model = SimpleCNN(num_classes=2)
        model.eval()

        dummy_input = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (4, 2)

    def test_get_model_baseline(self):
        """Test get_model factory function for baseline."""
        model = get_model("baseline", num_classes=2, dropout_rate=0.5)
        assert isinstance(model, BaselineCNN)

    def test_get_model_simple(self):
        """Test get_model factory function for simple."""
        model = get_model("simple", num_classes=2)
        assert isinstance(model, SimpleCNN)

    def test_get_model_invalid(self):
        """Test get_model with invalid model name."""
        with pytest.raises(ValueError):
            get_model("invalid_model")

    def test_count_parameters(self):
        """Test parameter counting function."""
        model = SimpleCNN(num_classes=2)
        param_count = count_parameters(model)

        assert isinstance(param_count, int)
        assert param_count > 0

        # Verify by manual counting
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count == manual_count

    def test_get_model_summary(self):
        """Test model summary function."""
        model = BaselineCNN(num_classes=2)
        summary = get_model_summary(model)

        assert isinstance(summary, dict)
        assert 'model_class' in summary
        assert 'total_parameters' in summary
        assert 'trainable_parameters' in summary
        assert 'model_size_mb' in summary

        assert summary['model_class'] == 'BaselineCNN'
        assert summary['total_parameters'] > 0
        assert summary['model_size_mb'] > 0

    def test_model_dropout_in_training_mode(self):
        """Test that dropout is active in training mode."""
        model = BaselineCNN(num_classes=2, dropout_rate=0.5)
        model.train()

        dummy_input = torch.randn(1, 3, 224, 224)

        # Multiple forward passes should give different results in training mode
        output1 = model(dummy_input)
        output2 = model(dummy_input)

        # Outputs should be different due to dropout
        # Note: There's a small chance they could be the same, but very unlikely
        assert not torch.allclose(output1, output2, atol=1e-6)

    def test_model_deterministic_in_eval_mode(self):
        """Test that model is deterministic in eval mode."""
        model = BaselineCNN(num_classes=2, dropout_rate=0.5)
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)

        # Multiple forward passes should give same results in eval mode
        with torch.no_grad():
            output1 = model(dummy_input)
            output2 = model(dummy_input)

        assert torch.allclose(output1, output2)

    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = BaselineCNN(num_classes=2, dropout_rate=0.5)
        model.train()

        dummy_input = torch.randn(2, 3, 224, 224)
        dummy_target = torch.tensor([0, 1])

        criterion = nn.CrossEntropyLoss()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)

        loss.backward()

        # Check that gradients exist for model parameters
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_model_cpu_gpu_compatibility(self):
        """Test model works on CPU (and GPU if available)."""
        model = BaselineCNN(num_classes=2)

        # Test on CPU
        model_cpu = model.to('cpu')
        dummy_input_cpu = torch.randn(1, 3, 224, 224).to('cpu')

        with torch.no_grad():
            output_cpu = model_cpu(dummy_input_cpu)

        assert output_cpu.device.type == 'cpu'

        # Test on GPU if available
        if torch.cuda.is_available():
            model_gpu = model.to('cuda')
            dummy_input_gpu = torch.randn(1, 3, 224, 224).to('cuda')

            with torch.no_grad():
                output_gpu = model_gpu(dummy_input_gpu)

            assert output_gpu.device.type == 'cuda'

    def test_batch_size_flexibility(self):
        """Test model works with different batch sizes."""
        model = BaselineCNN(num_classes=2)
        model.eval()

        batch_sizes = [1, 2, 4, 8, 16]

        for batch_size in batch_sizes:
            dummy_input = torch.randn(batch_size, 3, 224, 224)

            with torch.no_grad():
                output = model(dummy_input)

            assert output.shape == (batch_size, 2)

    def test_model_save_load(self):
        """Test model can be saved and loaded."""
        model = BaselineCNN(num_classes=2)
        dummy_input = torch.randn(1, 3, 224, 224)

        # Get output before saving
        model.eval()
        with torch.no_grad():
            output_before = model(dummy_input)

        # Save model
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
        torch.save(model.state_dict(), temp_file.name)
        temp_file.close()

        # Load model
        model_loaded = BaselineCNN(num_classes=2)
        model_loaded.load_state_dict(torch.load(temp_file.name))
        model_loaded.eval()

        # Get output after loading
        with torch.no_grad():
            output_after = model_loaded(dummy_input)

        # Outputs should be identical
        assert torch.allclose(output_before, output_after)

        # Cleanup
        os.unlink(temp_file.name)


class TestInferenceUtilities:
    """Test inference-related utility functions."""

    def test_model_prediction(self):
        """Test complete prediction pipeline."""
        model = get_model("baseline", num_classes=2)
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            logits = model(dummy_input)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)

        # Check outputs
        assert logits.shape == (1, 2)
        assert probabilities.shape == (1, 2)
        assert torch.allclose(probabilities.sum(dim=1), torch.tensor([1.0]))
        assert predicted_class.item() in [0, 1]

    def test_confidence_scores(self):
        """Test confidence score extraction."""
        model = get_model("baseline", num_classes=2)
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            logits = model(dummy_input)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # Confidence should be between 0 and 1
        assert 0 <= confidence.item() <= 1
        assert predicted.item() in [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
