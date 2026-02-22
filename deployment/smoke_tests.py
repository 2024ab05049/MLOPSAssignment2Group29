"""
Smoke tests for post-deployment validation.
Tests the deployed API to ensure it's working correctly.
"""

import os
import sys
import time
import requests
from PIL import Image
import io
import tempfile

# Configuration
API_URL = os.getenv('API_URL', 'http://localhost:8000')
MAX_RETRIES = 5
RETRY_DELAY = 10


class SmokeTestFailure(Exception):
    """Exception raised when a smoke test fails."""
    pass


def wait_for_service(url, max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Wait for the service to become available."""
    print(f"Waiting for service at {url} to become available...")

    for attempt in range(max_retries):
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✓ Service is available (attempt {attempt + 1}/{max_retries})")
                return True
        except requests.exceptions.RequestException as e:
            print(f"× Attempt {attempt + 1}/{max_retries} failed: {str(e)}")

        if attempt < max_retries - 1:
            print(f"  Waiting {delay} seconds before retry...")
            time.sleep(delay)

    return False


def test_health_endpoint():
    """Test the health check endpoint."""
    print("\n[TEST] Health Check Endpoint")
    print("-" * 50)

    try:
        response = requests.get(f"{API_URL}/health", timeout=10)

        if response.status_code != 200:
            raise SmokeTestFailure(f"Health check failed with status {response.status_code}")

        data = response.json()

        # Validate response structure
        required_fields = ['status', 'model_loaded', 'timestamp', 'version']
        for field in required_fields:
            if field not in data:
                raise SmokeTestFailure(f"Missing field in health response: {field}")

        # Check if model is loaded
        if not data.get('model_loaded'):
            raise SmokeTestFailure("Model is not loaded")

        print(f"✓ Status: {data['status']}")
        print(f"✓ Model loaded: {data['model_loaded']}")
        print(f"✓ Device: {data.get('device', 'unknown')}")
        print(f"✓ Version: {data['version']}")
        print("✓ Health check passed")

        return True

    except requests.exceptions.RequestException as e:
        raise SmokeTestFailure(f"Health check request failed: {str(e)}")


def test_root_endpoint():
    """Test the root endpoint."""
    print("\n[TEST] Root Endpoint")
    print("-" * 50)

    try:
        response = requests.get(f"{API_URL}/", timeout=10)

        if response.status_code != 200:
            raise SmokeTestFailure(f"Root endpoint failed with status {response.status_code}")

        data = response.json()

        if 'message' not in data or 'endpoints' not in data:
            raise SmokeTestFailure("Root endpoint returned invalid response")

        print(f"✓ Message: {data['message']}")
        print(f"✓ Endpoints: {list(data['endpoints'].keys())}")
        print("✓ Root endpoint passed")

        return True

    except requests.exceptions.RequestException as e:
        raise SmokeTestFailure(f"Root endpoint request failed: {str(e)}")


def test_prediction_endpoint():
    """Test the prediction endpoint with a dummy image."""
    print("\n[TEST] Prediction Endpoint")
    print("-" * 50)

    try:
        # Create a dummy image
        img = Image.new('RGB', (224, 224), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        # Make prediction request
        files = {'file': ('test_image.jpg', img_byte_arr, 'image/jpeg')}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)

        if response.status_code != 200:
            raise SmokeTestFailure(f"Prediction failed with status {response.status_code}: {response.text}")

        data = response.json()

        # Validate response structure
        required_fields = ['predicted_class', 'predicted_label', 'confidence', 'probabilities', 'inference_time_ms']
        for field in required_fields:
            if field not in data:
                raise SmokeTestFailure(f"Missing field in prediction response: {field}")

        # Validate predictions
        if data['predicted_class'] not in ['cat', 'dog']:
            raise SmokeTestFailure(f"Invalid predicted class: {data['predicted_class']}")

        if data['predicted_label'] not in [0, 1]:
            raise SmokeTestFailure(f"Invalid predicted label: {data['predicted_label']}")

        if not (0 <= data['confidence'] <= 1):
            raise SmokeTestFailure(f"Invalid confidence score: {data['confidence']}")

        # Check probabilities sum to 1
        probs_sum = sum(data['probabilities'].values())
        if not (0.99 <= probs_sum <= 1.01):
            raise SmokeTestFailure(f"Probabilities don't sum to 1: {probs_sum}")

        print(f"✓ Predicted class: {data['predicted_class']}")
        print(f"✓ Confidence: {data['confidence']:.4f}")
        print(f"✓ Inference time: {data['inference_time_ms']:.2f}ms")
        print(f"✓ Probabilities: {data['probabilities']}")
        print("✓ Prediction endpoint passed")

        return True

    except requests.exceptions.RequestException as e:
        raise SmokeTestFailure(f"Prediction request failed: {str(e)}")


def test_metrics_endpoint():
    """Test the metrics endpoint."""
    print("\n[TEST] Metrics Endpoint")
    print("-" * 50)

    try:
        response = requests.get(f"{API_URL}/metrics", timeout=10)

        if response.status_code != 200:
            raise SmokeTestFailure(f"Metrics endpoint failed with status {response.status_code}")

        # Check if response contains Prometheus metrics
        content = response.text
        if 'inference_requests_total' not in content:
            raise SmokeTestFailure("Metrics endpoint doesn't contain expected metrics")

        print("✓ Metrics endpoint is accessible")
        print(f"✓ Response length: {len(content)} bytes")
        print("✓ Metrics endpoint passed")

        return True

    except requests.exceptions.RequestException as e:
        raise SmokeTestFailure(f"Metrics request failed: {str(e)}")


def test_model_info_endpoint():
    """Test the model info endpoint."""
    print("\n[TEST] Model Info Endpoint")
    print("-" * 50)

    try:
        response = requests.get(f"{API_URL}/model-info", timeout=10)

        if response.status_code != 200:
            raise SmokeTestFailure(f"Model info endpoint failed with status {response.status_code}")

        data = response.json()

        if 'model_loaded' not in data or not data['model_loaded']:
            raise SmokeTestFailure("Model not loaded according to model-info endpoint")

        print(f"✓ Model loaded: {data['model_loaded']}")
        print(f"✓ Device: {data.get('device', 'unknown')}")
        print(f"✓ Class names: {data.get('class_names', [])}")
        print("✓ Model info endpoint passed")

        return True

    except requests.exceptions.RequestException as e:
        raise SmokeTestFailure(f"Model info request failed: {str(e)}")


def run_smoke_tests():
    """Run all smoke tests."""
    print("=" * 60)
    print("SMOKE TESTS - Cats vs Dogs Classifier API")
    print("=" * 60)
    print(f"API URL: {API_URL}")
    print()

    # Wait for service to be available
    if not wait_for_service(API_URL):
        print("\n" + "=" * 60)
        print("✗ SMOKE TESTS FAILED: Service not available")
        print("=" * 60)
        sys.exit(1)

    # Run tests
    tests = [
        ("Health Check", test_health_endpoint),
        ("Root Endpoint", test_root_endpoint),
        ("Prediction", test_prediction_endpoint),
        ("Metrics", test_metrics_endpoint),
        ("Model Info", test_model_info_endpoint)
    ]

    passed = 0
    failed = 0
    errors = []

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except SmokeTestFailure as e:
            failed += 1
            errors.append((test_name, str(e)))
            print(f"✗ {test_name} failed: {str(e)}")
        except Exception as e:
            failed += 1
            errors.append((test_name, f"Unexpected error: {str(e)}"))
            print(f"✗ {test_name} failed with unexpected error: {str(e)}")

    # Print summary
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed tests:")
        for test_name, error in errors:
            print(f"  - {test_name}: {error}")

    print("=" * 60)

    if failed > 0:
        print("\n✗ SMOKE TESTS FAILED")
        sys.exit(1)
    else:
        print("\n✓ ALL SMOKE TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    run_smoke_tests()
