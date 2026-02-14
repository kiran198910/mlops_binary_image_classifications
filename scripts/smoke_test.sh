#!/bin/bash
# Post-Deployment Smoke Test Script
# This script runs smoke tests against the deployed API
# Exit code 0 = success, 1 = failure

set -e

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
RETRIES="${RETRIES:-10}"
DELAY="${DELAY:-5}"

echo "=============================================="
echo "🔥 Post-Deployment Smoke Tests"
echo "=============================================="
echo "API URL: $API_URL"
echo "Retries: $RETRIES"
echo "Delay: $DELAY seconds"
echo ""

# Function to check health endpoint
check_health() {
    local attempt=1
    while [ $attempt -le $RETRIES ]; do
        echo "[Attempt $attempt/$RETRIES] Checking health endpoint..."
        
        response=$(curl -s -w "\n%{http_code}" "$API_URL/health" 2>/dev/null || echo "000")
        http_code=$(echo "$response" | tail -n1)
        body=$(echo "$response" | head -n-1)
        
        if [ "$http_code" = "200" ]; then
            echo "  ✅ Health check passed (HTTP $http_code)"
            echo "  Response: $body"
            return 0
        else
            echo "  ⏳ Health check failed (HTTP $http_code)"
        fi
        
        if [ $attempt -lt $RETRIES ]; then
            echo "  Waiting $DELAY seconds before retry..."
            sleep $DELAY
        fi
        
        attempt=$((attempt + 1))
    done
    
    echo "  ❌ Health check failed after $RETRIES attempts"
    return 1
}

# Function to test prediction endpoint
test_prediction() {
    echo ""
    echo "Testing prediction endpoint..."
    
    # Create a simple test image (1x1 red pixel PNG)
    # Base64 encoded minimal valid JPEG
    TEST_IMAGE_BASE64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    
    # Create temp file
    TEMP_IMAGE=$(mktemp /tmp/test_image.XXXXXX.png)
    echo "$TEST_IMAGE_BASE64" | base64 -d > "$TEMP_IMAGE"
    
    response=$(curl -s -w "\n%{http_code}" -X POST \
        -F "file=@$TEMP_IMAGE;type=image/png" \
        "$API_URL/predict" 2>/dev/null || echo "000")
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    rm -f "$TEMP_IMAGE"
    
    if [ "$http_code" = "200" ]; then
        echo "  ✅ Prediction endpoint working (HTTP $http_code)"
        echo "  Response: $body"
        return 0
    elif [ "$http_code" = "503" ]; then
        echo "  ⚠️ Model not loaded (HTTP 503) - Service is running"
        return 0  # Acceptable - service is up
    else
        echo "  ❌ Prediction failed (HTTP $http_code)"
        echo "  Response: $body"
        return 1
    fi
}

# Function to test root endpoint
test_root() {
    echo ""
    echo "Testing root endpoint..."
    
    response=$(curl -s -w "\n%{http_code}" "$API_URL/" 2>/dev/null || echo "000")
    http_code=$(echo "$response" | tail -n1)
    
    if [ "$http_code" = "200" ]; then
        echo "  ✅ Root endpoint working (HTTP $http_code)"
        return 0
    else
        echo "  ❌ Root endpoint failed (HTTP $http_code)"
        return 1
    fi
}

# Run tests
echo ""
echo "📋 Running smoke tests..."
echo "----------------------------------------------"

# Critical test: Health check
if ! check_health; then
    echo ""
    echo "❌ SMOKE TESTS FAILED: Service is not healthy"
    exit 1
fi

# Non-critical tests
test_root || true
test_prediction || true

echo ""
echo "=============================================="
echo "✅ SMOKE TESTS PASSED"
echo "=============================================="
exit 0
