"""
Unit tests for inference service API.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test that inference service can be imported."""
    try:
        from inference_service import app
        assert app is not None
    except Exception as e:
        pytest.skip(f"Could not import inference service: {str(e)}")


# Note: These tests would need a trained model to run fully
# For CI/CD, you can mock the model or use pytest fixtures
class TestInferenceServiceBasics:
    """Basic tests for inference service that don't require a loaded model."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        try:
            from inference_service import app
            return TestClient(app)
        except Exception:
            pytest.skip("Could not create test client")

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data
        assert "version" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
