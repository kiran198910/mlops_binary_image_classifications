# Makefile for Cats vs Dogs MLOps Project
# Provides convenient commands for common operations

# Python executable (change to python if needed)
PYTHON := python3

.PHONY: help install setup data train evaluate serve test lint docker-build docker-run clean

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║           Cats vs Dogs MLOps Project - Commands                  ║"
	@echo "╠══════════════════════════════════════════════════════════════════╣"
	@echo "║  make install      - Install dependencies                        ║"
	@echo "║  make setup        - Full setup (install + data)                 ║"
	@echo "║  make data         - Download and prepare dataset                ║"
	@echo "║  make data-sample  - Create sample dataset for testing           ║"
	@echo "║  make train        - Train the model                             ║"
	@echo "║  make evaluate     - Evaluate the model                          ║"
	@echo "║  make serve        - Start API server                            ║"
	@echo "║  make test         - Run tests                                   ║"
	@echo "║  make lint         - Run code linting                            ║"
	@echo "║  make format       - Format code with black                      ║"
	@echo "║  make docker-build - Build Docker image                          ║"
	@echo "║  make docker-run   - Run Docker container                        ║"
	@echo "║  make docker-train - Run training in Docker                      ║"
	@echo "║  make mlflow       - Start MLflow UI                             ║"
	@echo "║  make clean        - Clean generated files                       ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"

# ===========================================
# Installation
# ===========================================
install:
	@echo "Installing dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

setup: install data
	@echo "Setup complete!"

# ===========================================
# Data Operations
# ===========================================
data:
	@echo "Downloading and preparing dataset..."
	$(PYTHON) src/data/download_data.py --config config/config.yaml
	@echo "Dataset ready!"

data-sample:
	@echo "Creating sample dataset..."
	$(PYTHON) src/data/download_data.py --sample-only
	@echo "Sample dataset created!"

# ===========================================
# Model Operations
# ===========================================
train:
	@echo "Starting model training..."
	$(PYTHON) src/models/train.py --config config/config.yaml
	@echo "Training complete!"

train-quick:
	@echo "Running quick training (5 epochs)..."
	$(PYTHON) src/models/train.py --config config/config.yaml --epochs 5
	@echo "Quick training complete!"

evaluate:
	@echo "Evaluating model..."
	$(PYTHON) src/models/evaluate.py --config config/config.yaml
	@echo "Evaluation complete!"

# ===========================================
# API Server
# ===========================================
serve:
	@echo "Starting API server..."
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	@echo "Starting production API server..."
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# ===========================================
# Testing & Quality
# ===========================================
test:
	@echo "Running tests..."
	pytest tests/ -v --tb=short

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/"

lint:
	@echo "Running linters..."
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503
	@echo "Linting complete!"

format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/
	@echo "Formatting complete!"

# ===========================================
# Docker Operations
# ===========================================
docker-build:
	@echo "Building Docker image..."
	docker build -t cats-dogs-classifier:latest .
	@echo "Docker image built!"

docker-run:
	@echo "Running Docker container..."
	docker run -p 8000:8000 --rm cats-dogs-classifier:latest

docker-train:
	@echo "Running training in Docker..."
	docker-compose --profile training up training

docker-compose-up:
	@echo "Starting all services with Docker Compose..."
	docker-compose up -d api
	@echo "Services started!"

docker-compose-down:
	@echo "Stopping all services..."
	docker-compose down
	@echo "Services stopped!"

# ===========================================
# MLflow
# ===========================================
mlflow:
	@echo "Starting MLflow UI..."
	mlflow ui --host 0.0.0.0 --port 5000

mlflow-docker:
	@echo "Starting MLflow with Docker Compose..."
	docker-compose --profile tracking up -d mlflow
	@echo "MLflow server started at http://localhost:5000"

# ===========================================
# Smoke Tests
# ===========================================
smoke-test:
	@echo "Running smoke tests..."
	chmod +x scripts/smoke_test.sh
	API_URL=http://localhost:8000 RETRIES=5 DELAY=3 ./scripts/smoke_test.sh

smoke-test-python:
	@echo "Running Python smoke tests..."
	$(PYTHON) scripts/smoke_test.py --url http://localhost:8000 --retries 5 --delay 3

# ===========================================
# Cleanup
# ===========================================
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "Cleanup complete!"

clean-all: clean
	@echo "Deep cleaning (including data and models)..."
	rm -rf data/raw/* data/processed/* 2>/dev/null || true
	rm -rf models/checkpoints/* models/final/*.h5 2>/dev/null || true
	rm -rf logs/* mlruns/* evaluation_results/* 2>/dev/null || true
	@echo "Deep cleanup complete!"
