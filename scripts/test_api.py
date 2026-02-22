"""
Quick test script to verify API functionality locally.
Creates a test image and sends it to the API.
"""

import requests
from PIL import Image
import io
import sys
import time

API_URL = "http://localhost:8000"


def create_test_image():
    """Create a simple test image."""
    # Create a simple colored image
    img = Image.new('RGB', (224, 224), color=(254, 100, 100))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr


def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Health check passed")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check failed: {str(e)}")
        return False


def test_prediction():
    """Test prediction endpoint."""
    print("\nTesting prediction endpoint...")
    try:
        img = create_test_image()
        files = {'file': ('test.jpg', img, 'image/jpeg')}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=10)

        if response.status_code == 200:
            result = response.json()
            print("✓ Prediction successful")
            print(f"  Predicted class: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Inference time: {result['inference_time_ms']:.2f}ms")
            return True
        else:
            print(f"✗ Prediction failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Prediction failed: {str(e)}")
        return False


def test_metrics():
    """Test metrics endpoint."""
    print("\nTesting metrics endpoint...")
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        if response.status_code == 200:
            print("✓ Metrics endpoint accessible")
            return True
        else:
            print(f"✗ Metrics endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Metrics endpoint failed: {str(e)}")
        return False


def test_stats():
    """Test stats endpoint."""
    print("\nTesting stats endpoint...")
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            print("✓ Stats endpoint accessible")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ Stats endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Stats endpoint failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("API Local Test Suite")
    print("=" * 50)
    print(f"Testing API at: {API_URL}")
    print()

    # Wait for API to be ready
    print("Waiting for API to be ready...")
    for i in range(5):
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✓ API is ready")
                break
        except:
            pass
        time.sleep(2)
        print(f"  Retrying... ({i+1}/5)")
    else:
        print("✗ API not available. Make sure the service is running:")
        print("  python src/inference_service.py")
        sys.exit(1)

    # Run tests
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Prediction", test_prediction()))
    results.append(("Metrics", test_metrics()))
    results.append(("Stats", test_stats()))

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("=" * 50)

    if passed == total:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
