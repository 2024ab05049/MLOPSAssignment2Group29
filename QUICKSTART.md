# Quick Start Guide

Get the MLOps pipeline up and running in minutes!

## Prerequisites
- Python 3.10+
- pip
- Git

## 5-Minute Setup

### Step 1: Clone and Setup (1 min)
```bash
cd Assignment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Generate Sample Data (1 min)
```bash
python scripts/generate_sample_data.py --num_samples 100
```

### Step 3: Train Model (2 min - for sample data)
```bash
python src/train.py \
    --data_dir data/raw \
    --num_epochs 5 \
    --batch_size 16
```

### Step 4: Test API (1 min)
```bash
# Terminal 1: Start API
python src/inference_service.py

# Terminal 2: Test API
python scripts/test_api.py
```

## That's it! 🎉

Your MLOps pipeline is now running locally!

## What Next?

### View MLflow Experiments
```bash
mlflow ui --port 5000
# Open: http://localhost:5000
```

### Run Tests
```bash
pytest tests/ -v
```

### Build Docker Image
```bash
docker build -t catsdogs-api:latest .
docker run -p 8000:8000 -v $(pwd)/models:/app/models catsdogs-api:latest
```

### Deploy with Docker Compose
```bash
docker-compose up -d
# API: http://localhost:8000
# MLflow: http://localhost:5000
# Prometheus: http://localhost:9090
```

## Using Makefile (Recommended)

If you have `make` installed:

```bash
make help           # See all commands
make dev-setup      # Complete setup
make train          # Train model
make test           # Run tests
make serve          # Start API
make build          # Build Docker
make deploy         # Deploy with Docker Compose
```

## Test the API

### Health Check
```bash
curl http://localhost:8000/health
```

### Prediction (replace with your image path)
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/image.jpg"
```

### View Metrics
```bash
curl http://localhost:8000/metrics
curl http://localhost:8000/stats
```

## For Production Use

### 1. Use Real Dataset
Download Kaggle Cats vs Dogs dataset and place in `data/raw/`

### 2. Train Longer
```bash
python src/train.py --num_epochs 20 --batch_size 32
```

### 3. Deploy to Kubernetes
```bash
kubectl apply -f deployment/k8s/
python deployment/smoke_tests.py
```

## Troubleshooting

### Model Not Found
```bash
# Check if model exists
ls -lh models/best_model.pth

# If missing, train first
python src/train.py --data_dir data/raw --num_epochs 5
```

### Port Already in Use
```bash
# Change port (default: 8000)
PORT=8080 python src/inference_service.py
```

### Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

## Documentation

- **README.md**: Complete documentation
- **PROJECT_SUMMARY.md**: Implementation details
- **Makefile**: Quick command reference

## Support

For detailed instructions, see README.md
