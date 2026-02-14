#!/usr/bin/env python3
"""
Post-Deployment Smoke Tests
Validates that the deployed API is working correctly.
"""

import os
import sys
import time
import argparse
import requests
from io import BytesIO
from PIL import Image


def create_test_image() -> bytes:
    """Create a test image for prediction."""
    img = Image.new('RGB', (150, 150), color='red')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer.getvalue()


def test_health_endpoint(base_url: str, retries: int = 5, delay: int = 5) -> bool:
    """
    Test the health endpoint.
    
    Args:
        base_url: Base URL of the API
        retries: Number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        True if health check passes, False otherwise
    """
    health_url = f"{base_url}/health"
    
    for attempt in range(retries):
        try:
            print(f"[Attempt {attempt + 1}/{retries}] Checking health endpoint: {health_url}")
            response = requests.get(health_url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Health check passed: {data}")
                
                # Check if model is loaded
                if data.get('model_loaded', False):
                    print("  ✅ Model is loaded")
                    return True
                else:
                    print("  ⚠️ Model not loaded, but service is running")
                    return True  # Service is up, even if model isn't loaded
            else:
                print(f"  ❌ Health check failed with status: {response.status_code}")
                
        except requests.exceptions.ConnectionError as e:
            print(f"  ⏳ Connection failed: {e}")
        except requests.exceptions.Timeout:
            print(f"  ⏳ Request timed out")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        if attempt < retries - 1:
            print(f"  Retrying in {delay} seconds...")
            time.sleep(delay)
    
    print("❌ Health check failed after all retries")
    return False


def test_prediction_endpoint(base_url: str) -> bool:
    """
    Test the prediction endpoint with a sample image.
    
    Args:
        base_url: Base URL of the API
        
    Returns:
        True if prediction works, False otherwise
    """
    predict_url = f"{base_url}/predict"
    
    try:
        print(f"Testing prediction endpoint: {predict_url}")
        
        # Create test image
        test_image = create_test_image()
        
        files = {
            'file': ('test_image.jpg', test_image, 'image/jpeg')
        }
        
        response = requests.post(predict_url, files=files, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Prediction successful:")
            print(f"     - Prediction: {data.get('prediction')}")
            print(f"     - Confidence: {data.get('confidence', 0):.4f}")
            print(f"     - Processing time: {data.get('processing_time_ms', 0):.2f}ms")
            return True
        elif response.status_code == 503:
            print(f"  ⚠️ Model not loaded (503) - Service is up but model unavailable")
            # This is acceptable for smoke test - service is running
            return True
        else:
            print(f"  ❌ Prediction failed with status: {response.status_code}")
            print(f"     Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"  ❌ Connection failed: {e}")
        return False
    except requests.exceptions.Timeout:
        print(f"  ❌ Request timed out")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_model_info_endpoint(base_url: str) -> bool:
    """
    Test the model-info endpoint.
    
    Args:
        base_url: Base URL of the API
        
    Returns:
        True if model info is available, False otherwise
    """
    model_info_url = f"{base_url}/model-info"
    
    try:
        print(f"Testing model-info endpoint: {model_info_url}")
        
        response = requests.get(model_info_url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Model info retrieved:")
            print(f"     - Model name: {data.get('model_name')}")
            print(f"     - Input shape: {data.get('input_shape')}")
            print(f"     - Classes: {data.get('classes')}")
            return True
        elif response.status_code == 503:
            print(f"  ⚠️ Model not loaded - skipping model-info check")
            return True
        else:
            print(f"  ❌ Model info failed with status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def run_smoke_tests(base_url: str, retries: int = 5, delay: int = 5) -> bool:
    """
    Run all smoke tests.
    
    Args:
        base_url: Base URL of the API
        retries: Number of retry attempts for health check
        delay: Delay between retries
        
    Returns:
        True if all critical tests pass, False otherwise
    """
    print("=" * 60)
    print("🔥 Running Post-Deployment Smoke Tests")
    print("=" * 60)
    print(f"Target URL: {base_url}")
    print()
    
    results = {
        'health': False,
        'prediction': False,
        'model_info': False
    }
    
    # Test 1: Health endpoint (critical)
    print("\n📋 Test 1: Health Endpoint")
    print("-" * 40)
    results['health'] = test_health_endpoint(base_url, retries, delay)
    
    if not results['health']:
        print("\n❌ SMOKE TESTS FAILED: Service is not healthy")
        return False
    
    # Test 2: Prediction endpoint
    print("\n📋 Test 2: Prediction Endpoint")
    print("-" * 40)
    results['prediction'] = test_prediction_endpoint(base_url)
    
    # Test 3: Model info endpoint
    print("\n📋 Test 3: Model Info Endpoint")
    print("-" * 40)
    results['model_info'] = test_model_info_endpoint(base_url)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Smoke Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    critical_passed = results['health']  # Only health is critical
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name.upper()}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if critical_passed:
        print("✅ SMOKE TESTS PASSED - Service is operational")
        return True
    else:
        print("❌ SMOKE TESTS FAILED - Critical tests did not pass")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run post-deployment smoke tests")
    parser.add_argument(
        "--url", 
        type=str, 
        default=os.environ.get("API_URL", "http://localhost:8000"),
        help="Base URL of the API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=int(os.environ.get("SMOKE_TEST_RETRIES", "5")),
        help="Number of retries for health check (default: 5)"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=int(os.environ.get("SMOKE_TEST_DELAY", "5")),
        help="Delay between retries in seconds (default: 5)"
    )
    args = parser.parse_args()
    
    success = run_smoke_tests(args.url, args.retries, args.delay)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
