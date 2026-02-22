# Video Recording Guide - MLOps Pipeline Demo

## Prerequisites Check
Before recording, ensure you have:
- [ ] Python 3.10 or higher installed
- [ ] pip installed
- [ ] Git installed (optional, for showing version control)
- [ ] Docker Desktop installed (optional, for containerization demo)

## Setup Steps (Run these BEFORE recording)

### Step 1: Create Virtual Environment (2 minutes)
```cmd
cd c:\Assignment

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# You should see (venv) in your prompt
```

### Step 2: Install Dependencies (2-3 minutes)
```cmd
# Upgrade pip
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# This will take a few minutes - wait for completion
```

### Step 3: Create Directories
```cmd
# Create necessary directories
mkdir data\raw\cat
mkdir data\raw\dog
mkdir data\processed
mkdir models
mkdir logs
mkdir artifacts
```

### Step 4: Generate Sample Dataset (1 minute)
```cmd
# Generate 100 sample images (50 cats, 50 dogs)
python scripts\generate_sample_data.py --num_samples 50

# Verify data was created
dir data\raw\cat
dir data\raw\dog
```

### Step 5: Train the Model (5-10 minutes)
```cmd
# Train with reduced epochs for quick demo
python src\train.py --data_dir data\raw --num_epochs 5 --batch_size 16

# This will:
# - Create MLflow experiments
# - Train the model
# - Save best_model.pth in models/
# - Generate training curves and confusion matrix
```

### Step 6: Verify Model Created
```cmd
# Check model file exists
dir models\best_model.pth

# Should show the model file with size
```

---

## Video Recording Script (5 minutes)

### Scene 1: Project Overview (30 seconds)
**What to show:**
```cmd
# Show project structure
tree /F /A

# Or use dir command
dir

# Show README
type README.md | more
```

**What to say:**
"This is a complete end-to-end MLOps pipeline for binary image classification. It includes data versioning with DVC, model training with MLflow tracking, a FastAPI inference service, Docker containerization, and full CI/CD with GitHub Actions."

---

### Scene 2: MLflow Experiment Tracking (1 minute)
**What to do:**
```cmd
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Open browser to: http://localhost:5000
```

**What to show in browser:**
1. Navigate to the "cats_vs_dogs" experiment
2. Click on your run to show:
   - Parameters (batch_size, learning_rate, etc.)
   - Metrics (accuracy, loss, precision, recall)
   - Artifacts (confusion matrix, training curves)
3. Download and show the confusion matrix image
4. Show the training curves

**What to say:**
"Here's MLflow tracking our experiment. It logged all hyperparameters, metrics over each epoch, and artifacts like the confusion matrix and training curves. The model achieved X% accuracy on the validation set."

---

### Scene 3: Data Preprocessing & Model (45 seconds)
**What to show:**
```cmd
# Show data preprocessing code
type src\data_preprocessing.py | more

# Show model architecture
type src\model.py | more
```

**What to say:**
"The data preprocessing pipeline includes augmentation techniques like random flips and rotation. The baseline CNN has 4 convolutional blocks with batch normalization and dropout for regularization."

---

### Scene 4: API Testing (1 minute 15 seconds)
**What to do:**
```cmd
# Terminal 1: Start API server
python src\inference_service.py

# Terminal 2: Run tests
python scripts\test_api.py
```

**What to show:**
1. API starting up with logs
2. Health check passing
3. Prediction endpoint working
4. Response showing predicted class, confidence, and probabilities

**In browser (optional):**
- Open: http://localhost:8000
- Show: Interactive API docs at http://localhost:8000/docs
- Test prediction with image upload

**What to say:**
"The FastAPI service provides REST endpoints for health checks and predictions. It includes request logging, Prometheus metrics, and returns predictions with confidence scores. The inference time is around 40-50ms per image."

---

### Scene 5: Testing (30 seconds)
**What to do:**
```cmd
# Run unit tests
pytest tests\ -v --cov=src

# Show test results
```

**What to show:**
- Tests passing
- Coverage report showing >80%

**What to say:**
"We have comprehensive unit tests covering data preprocessing, model architecture, and API endpoints with over 80% code coverage."

---

### Scene 6: Docker & Containerization (45 seconds)
**What to do:**
```cmd
# Show Dockerfile
type Dockerfile

# Build Docker image (if Docker is installed)
docker build -t catsdogs-api:latest .

# Or show docker-compose
type docker-compose.yml
```

**What to say:**
"The service is containerized using Docker with a multi-stage build. The docker-compose file sets up the complete stack including the API, MLflow tracking server, and Prometheus for monitoring."

**If Docker is running:**
```cmd
# Start services
docker-compose up -d

# Show running containers
docker ps

# Test API
curl http://localhost:8000/health
```

---

### Scene 7: CI/CD Pipeline (1 minute)
**What to show:**
```cmd
# Show CI pipeline
type .github\workflows\ci.yml

# Show CD pipeline
type .github\workflows\cd.yml
```

**What to say:**
"The CI pipeline automatically runs tests, builds Docker images, and pushes to the container registry on every commit. The CD pipeline deploys to Kubernetes, runs smoke tests, and can automatically rollback on failure."

**Show:**
1. CI workflow structure (test → build → push)
2. CD workflow structure (deploy → smoke-test → rollback)
3. Smoke tests file

---

### Scene 8: Kubernetes Deployment (30 seconds)
**What to show:**
```cmd
# Show K8s deployment
type deployment\k8s\deployment.yaml

# Show service
type deployment\k8s\service.yaml

# Show smoke tests
type deployment\smoke_tests.py | more
```

**What to say:**
"For production deployment, we have Kubernetes manifests with proper health checks, resource limits, and horizontal pod autoscaling. Smoke tests validate the deployment after each update."

---

### Scene 9: Monitoring & Wrap-up (30 seconds)
**What to show:**
```cmd
# Show metrics endpoint
curl http://localhost:8000/metrics

# Show stats
curl http://localhost:8000/stats

# Show Prometheus config
type deployment\prometheus.yml
```

**What to say:**
"The service exposes Prometheus metrics for monitoring request counts, latency, and predictions by class. All requests are logged for debugging and performance tracking."

**Final statement:**
"This completes our end-to-end MLOps pipeline with data versioning, experiment tracking, containerization, automated testing, CI/CD, and production monitoring. Thank you!"

---

## Troubleshooting Common Issues

### Issue: Module not found
```cmd
# Make sure venv is activated
venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: MLflow UI not starting
```cmd
# Kill any existing MLflow processes
taskkill /F /IM mlflow.exe /T

# Restart
mlflow ui --host 0.0.0.0 --port 5000
```

### Issue: API port already in use
```cmd
# Use different port
set PORT=8080
python src\inference_service.py
```

### Issue: Model not found
```cmd
# Retrain the model
python src\train.py --data_dir data\raw --num_epochs 5 --batch_size 16
```

---

## Quick Commands Reference

### Train Model (Fast)
```cmd
python src\train.py --num_epochs 3 --batch_size 32
```

### Start API
```cmd
python src\inference_service.py
```

### Test API
```cmd
python scripts\test_api.py
```

### Run Tests
```cmd
pytest tests\ -v
```

### Start MLflow
```cmd
mlflow ui --port 5000
```

### Generate More Data
```cmd
python scripts\generate_sample_data.py --num_samples 100
```

---

## Recording Tips

1. **Clean your desktop** - Close unnecessary applications
2. **Increase font size** - Make terminal text readable
3. **Use split screen** - Show code and output side-by-side
4. **Practice first** - Do a dry run before recording
5. **Keep it smooth** - Don't pause too long, edit out mistakes
6. **Show results** - Always show the output of commands
7. **Time yourself** - Each section should fit in the allocated time
8. **Use zoom** - Zoom in on important parts
9. **Clear terminal** - Use `cls` between sections for clarity
10. **End strong** - Show the complete pipeline working

---

## Pre-Recording Checklist

- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] Sample data generated
- [ ] Model trained and saved
- [ ] API tested and working
- [ ] MLflow UI accessible
- [ ] Tests passing
- [ ] Terminal font size increased
- [ ] Screen recorder ready
- [ ] Microphone tested

---

## Post-Recording

1. Edit video to remove long waits
2. Add title slide with project name
3. Add ending slide with summary
4. Export as MP4
5. Check file size (<100MB if uploading)
6. If too large, use cloud storage link

---

## Alternative: Quick 3-Minute Demo

If you want a shorter demo:

1. **Overview** (20s) - Show project structure
2. **Training** (40s) - Show MLflow with results
3. **API** (40s) - Test health and prediction endpoints
4. **Docker** (30s) - Show docker-compose setup
5. **CI/CD** (30s) - Show workflow files
6. **Monitoring** (20s) - Show metrics endpoint

---

**Good luck with your video recording!** 🎥🚀
