# MLOps Pipeline for Cats vs Dogs Classification

Complete end-to-end MLOps pipeline for binary image classification (Cats vs Dogs) for a pet adoption platform.

## Project Overview

This project implements a production-ready MLOps pipeline including:
- ✅ **M1**: Model development with experiment tracking (MLflow)
- ✅ **M2**: Model packaging and containerization (Docker)
- ✅ **M3**: CI pipeline for automated testing and building (GitHub Actions)
- ✅ **M4**: CD pipeline with Kubernetes deployment
- ✅ **M5**: Monitoring, logging, and metrics collection (Prometheus)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Data] → [Training + MLflow] → [Model] → [Docker] → [K8s]     │
│                                                                  │
│  └─ DVC    └─ Experiment Tracking  └─ API    └─ CI/CD          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Data Versioning**: DVC for dataset tracking
- **Experiment Tracking**: MLflow for logging metrics, parameters, and artifacts
- **Model Training**: PyTorch-based CNN with data augmentation
- **REST API**: FastAPI service with health checks and prediction endpoints
- **Containerization**: Multi-stage Dockerfile for production deployment
- **CI/CD**: GitHub Actions for automated testing, building, and deployment
- **Monitoring**: Prometheus metrics and request logging
- **Kubernetes**: Production-ready K8s manifests with HPA
- **Testing**: Comprehensive unit tests and smoke tests

## Project Structure

```
Assignment/
├── .github/
│   └── workflows/
│       ├── ci.yml              # CI pipeline
│       └── cd.yml              # CD pipeline
├── .dvc/                       # DVC configuration
├── deployment/
│   ├── k8s/
│   │   ├── deployment.yaml    # K8s deployment
│   │   ├── service.yaml       # K8s service
│   │   └── configmap-hpa.yaml # ConfigMap & HPA
│   ├── smoke_tests.py         # Post-deployment tests
│   └── prometheus.yml         # Metrics configuration
├── src/
│   ├── data_preprocessing.py  # Data loading & augmentation
│   ├── model.py               # CNN architecture
│   ├── train.py               # Training script with MLflow
│   └── inference_service.py   # FastAPI application
├── tests/
│   ├── test_preprocessing.py  # Data preprocessing tests
│   ├── test_model.py          # Model tests
│   └── test_inference_service.py  # API tests
├── models/                     # Saved models
├── data/
│   ├── raw/                   # Original dataset
│   │   ├── cat/
│   │   └── dog/
│   └── processed/             # Preprocessed data
├── mlruns/                    # MLflow artifacts
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Local deployment
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Kubernetes cluster (minikube/kind/microk8s for local)
- Git & DVC
- NVIDIA GPU (optional, for faster training)

## Setup Instructions

### 1. Clone and Setup Environment

```bash
# Clone repository
git clone <your-repo-url>
cd Assignment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download the Cats and Dogs dataset from Kaggle:

```bash
# Create data directory structure
mkdir -p data/raw/cat data/raw/dog

# Download and extract dataset
# Place cat images in data/raw/cat/
# Place dog images in data/raw/dog/

# Initialize DVC
dvc init
dvc add data/raw
git add data/raw.dvc .dvc/config
git commit -m "Add dataset with DVC"
```

### 3. Train Model (M1)

```bash
# Start MLflow UI (optional, for viewing experiments)
mlflow ui --host 0.0.0.0 --port 5000

# Train model with MLflow tracking
python src/train.py \
    --data_dir data/raw \
    --model_name baseline \
    --num_epochs 10 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --experiment_name cats_vs_dogs \
    --run_name baseline_run_1

# View MLflow UI at http://localhost:5000
```

**Training outputs:**
- Trained model: `models/best_model.pth`
- MLflow artifacts: `mlruns/`
- Training curves, confusion matrix, and metrics

### 4. Test Locally (M2)

```bash
# Run inference service locally
python src/inference_service.py

# In another terminal, test the API
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/test/image.jpg"
```

### 5. Build Docker Image (M2)

```bash
# Build Docker image
docker build -t catsdogs-api:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  catsdogs-api:latest

# Test using curl
curl http://localhost:8000/health
```

### 6. Deploy with Docker Compose

```bash
# Start all services (API + MLflow + Prometheus)
docker-compose up -d

# Check services
docker-compose ps

# View logs
docker-compose logs -f api

# Test API
curl http://localhost:8000/health

# Access services
# - API: http://localhost:8000
# - MLflow: http://localhost:5000
# - Prometheus: http://localhost:9090

# Stop services
docker-compose down
```

### 7. Run Tests (M3)

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py -v

# View coverage report in browser
open htmlcov/index.html
```

### 8. Deploy to Kubernetes (M4)

```bash
# Start local Kubernetes cluster (if using minikube)
minikube start

# Create namespace
kubectl create namespace mlops

# Apply Kubernetes manifests
kubectl apply -f deployment/k8s/ -n mlops

# Check deployment status
kubectl get pods -n mlops
kubectl get svc -n mlops

# Port forward to access service locally
kubectl port-forward svc/catsdogs-api-service 8000:80 -n mlops

# Test the deployed service
curl http://localhost:8000/health
```

### 9. Run Smoke Tests (M4)

```bash
# Set API URL
export API_URL=http://localhost:8000

# Run smoke tests
python deployment/smoke_tests.py
```

### 10. Monitoring (M5)

```bash
# Access Prometheus (if using docker-compose)
open http://localhost:9090

# Query metrics
# - inference_requests_total
# - inference_request_latency_seconds
# - predictions_total

# View API metrics endpoint
curl http://localhost:8000/metrics

# View API statistics
curl http://localhost:8000/stats

# View recent logs
curl http://localhost:8000/logs?limit=50
```

## CI/CD Pipeline

### Continuous Integration (M3)

The CI pipeline (`.github/workflows/ci.yml`) runs on every push/PR:

1. **Test Stage**:
   - Checkout code
   - Install dependencies
   - Run unit tests with pytest
   - Generate coverage report

2. **Build Stage**:
   - Build Docker image
   - Push to GitHub Container Registry
   - Tag with commit SHA and branch name

3. **Security Stage**:
   - Scan image with Trivy
   - Upload results to GitHub Security

### Continuous Deployment (M4)

The CD pipeline (`.github/workflows/cd.yml`) runs on main branch:

1. **Deploy Stage**:
   - Update Kubernetes manifests
   - Apply to cluster
   - Wait for rollout

2. **Test Stage**:
   - Run smoke tests
   - Verify endpoints

3. **Rollback Stage** (on failure):
   - Automatically rollback to previous version
   - Notify team

## API Endpoints

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "timestamp": "2024-01-15T10:30:00",
  "version": "1.0.0"
}
```

### Prediction
```bash
POST /predict
Content-Type: multipart/form-data
```

Request:
- `file`: Image file (JPG, PNG)

Response:
```json
{
  "predicted_class": "dog",
  "predicted_label": 1,
  "confidence": 0.9876,
  "probabilities": {
    "cat": 0.0124,
    "dog": 0.9876
  },
  "inference_time_ms": 45.23,
  "timestamp": "2024-01-15T10:30:00"
}
```

### Metrics
```bash
GET /metrics
```

Returns Prometheus-formatted metrics.

### Statistics
```bash
GET /stats
```

Response:
```json
{
  "total_requests": 1234,
  "predictions_by_class": {
    "cat": 567,
    "dog": 667
  },
  "average_inference_time_ms": 42.5,
  "average_confidence": 0.943
}
```

### Model Info
```bash
GET /model-info
```

Response:
```json
{
  "model_loaded": true,
  "device": "cpu",
  "class_names": ["cat", "dog"],
  "metadata": {
    "model_name": "baseline",
    "best_val_acc": 0.95,
    "total_parameters": 1234567
  }
}
```

## Model Performance

Expected performance (after training on full dataset):

- **Accuracy**: ~94-96%
- **Precision**: ~95%
- **Recall**: ~94%
- **F1-Score**: ~95%
- **Inference Time**: ~40-50ms (CPU), ~5-10ms (GPU)

## Configuration

### Environment Variables

- `MODEL_PATH`: Path to model file (default: `/app/models/best_model.pth`)
- `PORT`: API port (default: `8000`)
- `API_URL`: API URL for smoke tests (default: `http://localhost:8000`)

### Training Hyperparameters

Can be customized via command-line arguments:

```bash
python src/train.py \
  --model_name baseline \
  --batch_size 32 \
  --num_epochs 20 \
  --learning_rate 0.001 \
  --dropout 0.5 \
  --weight_decay 1e-4
```

## Monitoring & Logging

### Request Logging

All requests are logged with:
- Timestamp
- Filename
- Predicted class
- Confidence score
- Inference time

Access logs via `/logs` endpoint.

### Prometheus Metrics

Key metrics collected:
- `inference_requests_total{endpoint, status}`: Total requests
- `inference_request_latency_seconds{endpoint}`: Request latency
- `predictions_total{predicted_class}`: Predictions by class

### Model Performance Tracking

Post-deployment monitoring includes:
- Request/response logging (excluding sensitive data)
- Prediction distribution tracking
- Inference time monitoring
- Confidence score distribution

## Troubleshooting

### Model not loading

```bash
# Check model file exists
ls -lh models/best_model.pth

# Check model path in environment
echo $MODEL_PATH

# Test model loading
python -c "from src.inference_service import load_model; load_model()"
```

### Container issues

```bash
# Check Docker logs
docker logs <container-id>

# Rebuild without cache
docker build --no-cache -t catsdogs-api:latest .

# Check health
docker exec <container-id> curl http://localhost:8000/health
```

### Kubernetes issues

```bash
# Check pod status
kubectl describe pod <pod-name> -n mlops

# View pod logs
kubectl logs <pod-name> -n mlops

# Check events
kubectl get events -n mlops --sort-by='.lastTimestamp'
```

## Development Workflow

1. **Feature Development**:
   - Create feature branch
   - Implement changes
   - Write/update tests
   - Push to trigger CI

2. **Model Training**:
   - Train with MLflow tracking
   - Compare experiments in MLflow UI
   - Select best model
   - Update model file

3. **Deployment**:
   - Merge to main branch
   - CI builds and pushes Docker image
   - CD deploys to Kubernetes
   - Smoke tests validate deployment

## Best Practices

- ✅ Version control all code with Git
- ✅ Version control data with DVC
- ✅ Track all experiments with MLflow
- ✅ Write unit tests for all functions
- ✅ Use CI/CD for automated deployment
- ✅ Monitor model performance in production
- ✅ Log all predictions for debugging
- ✅ Use health checks for service reliability

