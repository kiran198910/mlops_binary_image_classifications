"""
Unit Tests for Cats vs Dogs MLOps Project
"""

import os
import sys
import pytest
import numpy as np
from PIL import Image
import io
import tempfile

# Add src to paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDataPreprocessing:
    """Tests for data preprocessing functions."""
    
    def test_preprocess_image_bytes(self):
        """Test preprocessing image from bytes."""
        from src.data.preprocessing import preprocess_image_bytes, load_config
        
        # Create a dummy config
        config = {
            'data': {'image_size': 150}
        }
        
        # Create a test image
        img = Image.new('RGB', (200, 200), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Preprocess
        result = preprocess_image_bytes(img_bytes.getvalue(), config)
        
        # Assertions
        assert result.shape == (1, 150, 150, 3)
        assert result.dtype == np.float64
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_verify_image(self):
        """Test image verification function."""
        from src.data.preprocessing import verify_image
        
        # Create a valid temporary image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img = Image.new('RGB', (100, 100), color='blue')
            img.save(f.name)
            
            assert verify_image(f.name) == True
            os.unlink(f.name)
    
    def test_verify_invalid_image(self):
        """Test image verification with invalid file."""
        from src.data.preprocessing import verify_image
        
        # Create an invalid image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(b'not an image')
            f.flush()
            
            assert verify_image(f.name) == False
            os.unlink(f.name)


class TestModelBuilding:
    """Tests for model building functions."""
    
    def test_build_custom_cnn(self):
        """Test building custom CNN model."""
        from src.models.model import build_custom_cnn
        
        model = build_custom_cnn((150, 150, 3), 2)
        
        assert model is not None
        assert model.input_shape == (None, 150, 150, 3)
        assert model.output_shape == (None, 1)
    
    def test_compile_model(self):
        """Test model compilation."""
        from src.models.model import build_custom_cnn, compile_model
        
        config = {
            'training': {
                'learning_rate': 0.001,
                'optimizer': 'adam'
            }
        }
        
        model = build_custom_cnn((150, 150, 3), 2)
        compiled_model = compile_model(model, config)
        
        assert compiled_model.optimizer is not None
        assert compiled_model.loss is not None
    
    def test_get_model(self):
        """Test getting model from config."""
        from src.models.model import get_model
        
        config = {
            'model': {
                'architecture': 'custom_cnn',
                'input_shape': [150, 150, 3],
                'num_classes': 2
            }
        }
        
        model = get_model(config)
        assert model is not None


class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from src.api.main import app
        
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data
    
    def test_predict_no_file(self, client):
        """Test predict endpoint without file."""
        response = client.post("/predict")
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_file_type(self, client):
        """Test predict endpoint with invalid file type."""
        # Create a text file
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/predict", files=files)
        # Should return error (either 400 or 503 if model not loaded)
        assert response.status_code in [400, 503]


class TestConfigLoading:
    """Tests for configuration loading."""
    
    def test_load_config(self):
        """Test loading configuration file."""
        import yaml
        
        config_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'config', 
            'config.yaml'
        )
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert 'data' in config
            assert 'model' in config
            assert 'training' in config
            assert 'inference' in config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
