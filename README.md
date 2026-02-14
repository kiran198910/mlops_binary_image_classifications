# 🐱🐶 Cats vs Dogs MLOps Project

A complete end-to-end MLOps pipeline for image classification of cats and dogs. This project demonstrates industry best practices for machine learning operations including data pipelines, model training, experiment tracking, API deployment, containerization, and CI/CD.

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
- [API Usage](#api-usage)
- [Docker Deployment](#docker-deployment)
- [MLflow Tracking](#mlflow-tracking)
- [CI/CD Pipeline](#cicd-pipeline)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **Data Pipeline**: Automated data download, preprocessing, and train/val/test splitting
- **Model Training**: CNN-based image classification with multiple architecture options
- **Experiment Tracking**: MLflow integration for metrics, parameters, and model versioning
- **REST API**: FastAPI-powered inference server with batch prediction support
- **Containerization**: Docker and Docker Compose for consistent deployments
- **CI/CD**: GitHub Actions workflow for automated testing and deployment
- **Code Quality**: Integrated linting, formatting, and testing

## 📁 Project Structure

```
mlops-cats-dogs/
├── config/
│   └── config.yaml            # Main configuration file
├── data/
│   ├── raw/                   # Downloaded raw data
│   └── processed/             # Preprocessed train/val/test data
├── models/
│   ├── checkpoints/           # Training checkpoints
│   └── final/                 # Final trained models
├── src/
│   ├── api/
│   │   └── main.py            # FastAPI inference server
│   ├── data/
│   │   ├── download_data.py   # Data download and preparation
│   │   └── preprocessing.py   # Image preprocessing utilities
│   └── models/
│       ├── model.py           # Model architectures
│       ├── train.py           # Training script
│       └── evaluate.py        # Evaluation script
├── tests/
│   └── test_mlops.py          # Unit tests
├── .github/
│   └── workflows/
│       └── ci-cd.yaml         # GitHub Actions workflow
├── Dockerfile                 # Production Dockerfile
├── Dockerfile.train           # Training Dockerfile
├── docker-compose.yaml        # Docker Compose configuration
├── Makefile                   # Convenience commands
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)
- Git

### Option 1: Local Setup (Recommended for Development)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/mlops-cats-dogs.git
cd mlops-cats-dogs

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
make install
# OR: pip install -r requirements.txt

# 4. Prepare sample dataset (for quick testing)
make data-sample

# 5. Train the model
make train-quick  # Quick training with 5 epochs

# 6. Start the API server
make serve
```

### Option 2: Docker Setup (Recommended for Production)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/mlops-cats-dogs.git
cd mlops-cats-dogs

# 2. Build and run with Docker Compose
docker-compose up -d api

# The API will be available at http://localhost:8000
```

## 📖 Step-by-Step Guide

### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/MacOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Data Preparation

**Option A: Download Full Dataset (Microsoft Cats & Dogs)**
```bash
python src/data/download_data.py
```

**Option B: Create Sample Dataset (For Testing)**
```bash
python src/data/download_data.py --sample-only
```

This will:
- Download/create images in `data/raw/`
- Split data into `data/processed/train/`, `validation/`, and `test/`

### Step 3: Model Training

**Basic Training:**
```bash
python src/models/train.py
```

**Training with Custom Parameters:**
```bash
python src/models/train.py --epochs 30 --learning-rate 0.0001
```

**Training without MLflow:**
```bash
python src/models/train.py --no-mlflow
```

The trained model will be saved to `models/final/cats_dogs_model.h5`

### Step 4: Model Evaluation

```bash
python src/models/evaluate.py
```

This generates:
- Confusion matrix (`evaluation_results/confusion_matrix.png`)
- ROC curve (`evaluation_results/roc_curve.png`)
- Precision-Recall curve (`evaluation_results/pr_curve.png`)
- Classification report (`evaluation_results/evaluation_results.json`)

### Step 5: Start API Server

**Development Mode (with auto-reload):**
```bash
make serve
# OR: python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Production Mode:**
```bash
make serve-prod
# OR: python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Step 6: Test the API

Access the API documentation at: `http://localhost:8000/docs`

**Using curl:**
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

**Using Python:**
```python
import requests

# Single prediction
with open("cat.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
print(response.json())
```

## 🌐 API Usage

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API documentation (HTML) |
| `/health` | GET | Health check |
| `/model-info` | GET | Model information |
| `/predict` | POST | Single image prediction |
| `/predict/batch` | POST | Batch prediction (max 10 images) |
| `/docs` | GET | Swagger UI documentation |

### Example Response

```json
{
  "prediction": "cat",
  "confidence": 0.9234,
  "class_probabilities": {
    "cat": 0.9234,
    "dog": 0.0766
  },
  "processing_time_ms": 45.23,
  "timestamp": "2026-02-12T10:30:00.000000"
}
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
# Build production image
docker build -t cats-dogs-classifier:latest .

# Run container
docker run -p 8000:8000 cats-dogs-classifier:latest
```

### Using Docker Compose

```bash
# Start API service
docker-compose up -d api

# Start with MLflow tracking
docker-compose --profile tracking up -d

# Start training service
docker-compose --profile training up training

# Stop all services
docker-compose down
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHONUNBUFFERED` | Python output buffering | 1 |
| `MLFLOW_TRACKING_URI` | MLflow server URI | `mlruns` |

## 📊 MLflow Tracking

### Start MLflow UI

```bash
# Local
mlflow ui --host 0.0.0.0 --port 5000

# With Docker Compose
docker-compose --profile tracking up -d mlflow
```

Access MLflow UI at: `http://localhost:5000`

### Tracked Metrics
- Training/Validation Accuracy
- Training/Validation Loss
- Precision, Recall, F1-Score
- ROC-AUC Score

## 🔄 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yaml`) automates:

1. **Linting**: Code quality checks with Black, isort, Flake8
2. **Testing**: Unit tests with pytest
3. **Building**: Docker image creation
4. **Pushing**: Push to Docker registry (on main branch)
5. **Deployment**: Deploy to production (configurable)

### Setup CI/CD

1. Go to repository Settings → Secrets
2. Add the following secrets:
   - `DOCKER_USERNAME`: Docker Hub username
   - `DOCKER_PASSWORD`: Docker Hub password/token

### Manual Training Trigger

```yaml
# Trigger training via GitHub Actions
workflow_dispatch:
  inputs:
    run_training: true
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
data:
  image_size: 150        # Image dimensions
  batch_size: 32         # Training batch size
  validation_split: 0.2  # Validation data ratio

model:
  architecture: custom_cnn  # Options: custom_cnn, mobilenet, resnet50

training:
  epochs: 20
  learning_rate: 0.001
  optimizer: adam

inference:
  threshold: 0.5
  port: 8000
```

## 🧪 Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_mlops.py -v
```

## 📝 Available Make Commands

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make setup` | Full setup (install + data) |
| `make data` | Download and prepare dataset |
| `make data-sample` | Create sample dataset |
| `make train` | Train model (full) |
| `make train-quick` | Train model (5 epochs) |
| `make evaluate` | Evaluate model |
| `make serve` | Start API server (dev) |
| `make serve-prod` | Start API server (prod) |
| `make test` | Run tests |
| `make lint` | Run linting |
| `make format` | Format code |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run Docker container |
| `make mlflow` | Start MLflow UI |
| `make clean` | Clean generated files |

## 🔧 Troubleshooting

### Common Issues

**1. Model not loading**
```bash
# Ensure model file exists
ls -la models/final/cats_dogs_model.h5

# If not, train the model first
make train-quick
```

**2. Out of memory during training**
```yaml
# Reduce batch size in config/config.yaml
data:
  batch_size: 16  # Reduce from 32
```

**3. Docker build fails**
```bash
# Clear Docker cache
docker system prune -a

# Rebuild
docker build --no-cache -t cats-dogs-classifier:latest .
```

**4. API returns 503 (Model not loaded)**
- Ensure model file exists at `models/final/cats_dogs_model.h5`
- Check the path in `config/config.yaml`

**5. Permission denied errors**
```bash
# Linux: Fix permissions
chmod -R 755 data/ models/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Happy Classifying! 🐱🐶**
