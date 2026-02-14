"""
FastAPI Inference Server for Cats vs Dogs Classifier
"""

import os
import io
import sys
from typing import Dict, List, Optional
from datetime import datetime

import yaml
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from pydantic import BaseModel

# Load configuration
def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    # Try multiple paths for config
    possible_paths = [
        config_path,
        "../config/config.yaml",
        "../../config/config.yaml",
        os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f)
    
    # Return default config if file not found
    return {
        'data': {'image_size': 150},
        'inference': {
            'model_path': 'models/final/cats_dogs_model.h5',
            'threshold': 0.5,
            'host': '0.0.0.0',
            'port': 8000
        },
        'model': {'classes': ['cat', 'dog']}
    }


# Initialize FastAPI app
app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="MLOps API for classifying images of cats and dogs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and config
model = None
config = None


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    prediction: str
    confidence: float
    class_probabilities: Dict[str, float]
    processing_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Response model for model info."""
    model_name: str
    input_shape: List[int]
    classes: List[str]
    threshold: float


def load_model_on_startup():
    """Load the model at startup."""
    global model, config
    
    config = load_config()
    model_path = config['inference']['model_path']
    
    # Try multiple paths
    possible_paths = [
        model_path,
        os.path.join(os.path.dirname(__file__), "..", "..", model_path),
        f"/app/{model_path}"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading model from {path}...")
            model = tf.keras.models.load_model(path)
            print("Model loaded successfully!")
            return True
    
    print(f"Warning: Model not found. Tried paths: {possible_paths}")
    return False


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    load_model_on_startup()


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for model inference.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Preprocessed numpy array
    """
    image_size = config['data']['image_size']
    
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize((image_size, image_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cats vs Dogs Classifier API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            code { background: #e0e0e0; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>🐱🐶 Cats vs Dogs Classifier API</h1>
        <p>Welcome to the Cats vs Dogs image classification API!</p>
        
        <h2>Available Endpoints:</h2>
        
        <div class="endpoint">
            <h3>POST /predict</h3>
            <p>Upload an image to get a prediction.</p>
            <p>Request: <code>multipart/form-data</code> with <code>file</code> field</p>
        </div>
        
        <div class="endpoint">
            <h3>GET /health</h3>
            <p>Check API health status.</p>
        </div>
        
        <div class="endpoint">
            <h3>GET /model-info</h3>
            <p>Get information about the loaded model.</p>
        </div>
        
        <div class="endpoint">
            <h3>GET /docs</h3>
            <p>Interactive API documentation (Swagger UI).</p>
        </div>
        
        <p><a href="/docs">📚 View Full API Documentation</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_name="Cats vs Dogs Classifier",
        input_shape=list(model.input_shape[1:]),
        classes=config['model'].get('classes', ['cat', 'dog']),
        threshold=config['inference']['threshold']
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict whether an image contains a cat or dog.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results with confidence scores
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure the model file exists."
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image."
        )
    
    start_time = datetime.now()
    
    try:
        # Read and preprocess image
        contents = await file.read()
        img_array = preprocess_image(contents)
        
        # Make prediction
        prediction_prob = model.predict(img_array, verbose=0)[0][0]
        
        # Determine class
        threshold = config['inference']['threshold']
        predicted_class = 'dog' if prediction_prob > threshold else 'cat'
        confidence = prediction_prob if predicted_class == 'dog' else 1 - prediction_prob
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            prediction=predicted_class,
            confidence=float(confidence),
            class_probabilities={
                'cat': float(1 - prediction_prob),
                'dog': float(prediction_prob)
            },
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict for multiple images at once.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        List of prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images per batch request"
        )
    
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            img_array = preprocess_image(contents)
            
            prediction_prob = model.predict(img_array, verbose=0)[0][0]
            threshold = config['inference']['threshold']
            predicted_class = 'dog' if prediction_prob > threshold else 'cat'
            confidence = prediction_prob if predicted_class == 'dog' else 1 - prediction_prob
            
            results.append({
                'filename': file.filename,
                'prediction': predicted_class,
                'confidence': float(confidence),
                'class_probabilities': {
                    'cat': float(1 - prediction_prob),
                    'dog': float(prediction_prob)
                }
            })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return {'predictions': results, 'total': len(results)}


if __name__ == "__main__":
    import uvicorn
    
    config = load_config()
    host = config['inference']['host']
    port = config['inference']['port']
    
    print(f"Starting server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
