# Video Recording Checklist ✓

## Phase 1: Initial Setup (Do this FIRST - 5-10 minutes)

### Step 1: Run Automated Setup
```cmd
cd c:\Assignment
SETUP_FOR_VIDEO.bat
```
This will:
- ✓ Create virtual environment
- ✓ Install all dependencies
- ✓ Create necessary directories
- ✓ Generate sample dataset (100 images)

**Wait for it to complete** - Takes 3-5 minutes

---

### Step 2: Train the Model
```cmd
TRAIN_FOR_VIDEO.bat
```
This will:
- ✓ Train model for 5 epochs (~5-10 minutes)
- ✓ Start MLflow UI automatically
- ✓ Save model to models/best_model.pth
- ✓ Generate metrics and artifacts

**Wait for training to complete** - Takes 5-10 minutes

---

### Step 3: Verify Everything Works
```cmd
TEST_BEFORE_VIDEO.bat
```
This will:
- ✓ Check all packages installed
- ✓ Verify data exists
- ✓ Verify model trained
- ✓ Run unit tests
- ✓ Test preprocessing and model loading

**Should show all PASS** - Takes 1-2 minutes

---

## Phase 2: Prepare for Recording (5 minutes)

### Environment Preparation
- [ ] Close all unnecessary applications
- [ ] Clear browser cache/history
- [ ] Clean desktop (remove personal files/icons)
- [ ] Close personal emails, chat applications
- [ ] Turn off notifications (Windows Focus Assist)
- [ ] Close auto-updating apps

### Terminal Setup
- [ ] Open Command Prompt or PowerShell
- [ ] Set large font size (View → Zoom → Large)
- [ ] Set dark/light theme (your preference)
- [ ] Clear terminal: `cls`
- [ ] CD to project: `cd c:\Assignment`
- [ ] Activate venv: `venv\Scripts\activate`

### Browser Setup
- [ ] Open Chrome/Edge in new window
- [ ] Clear browsing data
- [ ] Zoom level: 110-125% (make text readable)
- [ ] Bookmark these URLs:
  - http://localhost:5000 (MLflow)
  - http://localhost:8000 (API)
  - http://localhost:8000/docs (API Docs)

### Recording Software
- [ ] Screen recorder installed (OBS, Camtasia, or built-in Game Bar)
- [ ] Set recording area (full screen or window)
- [ ] Set video quality (1080p recommended)
- [ ] Test microphone
- [ ] Test audio levels
- [ ] Do a 10-second test recording

---

## Phase 3: Recording Session (5 minutes)

### Have These Files Open in Tabs:
1. VIDEO_RECORDING_GUIDE.md (your script)
2. Terminal 1 (for main commands)
3. Terminal 2 (for API testing)
4. Browser (for MLflow and API docs)

### Recording Flow:

#### Scene 1: Introduction (30 seconds)
**Terminal:**
```cmd
cd c:\Assignment
dir
type README.md
```
**Say:** "Welcome to our end-to-end MLOps pipeline for cats vs dogs classification..."

#### Scene 2: MLflow Tracking (1 minute)
**Browser:**
- Open http://localhost:5000
- Click on "cats_vs_dogs" experiment
- Show run details, metrics, artifacts
**Say:** "MLflow tracked all our experiments. Here's the accuracy, precision, and training curves..."

#### Scene 3: Code Overview (45 seconds)
**Terminal:**
```cmd
type src\data_preprocessing.py
type src\model.py
```
**Say:** "The preprocessing pipeline includes augmentation, and our CNN has 4 convolutional blocks..."

#### Scene 4: API Demo (1 minute 15 seconds)
**Terminal 1:**
```cmd
python src\inference_service.py
```

**Terminal 2:**
```cmd
python scripts\test_api.py
```

**Browser:**
- Open http://localhost:8000
- Show API docs
- Test prediction

**Say:** "The FastAPI service provides health checks, predictions with confidence scores, and Prometheus metrics..."

#### Scene 5: Testing (30 seconds)
**Terminal:**
```cmd
pytest tests\ -v --cov=src
```
**Say:** "We have comprehensive tests with over 80% coverage..."

#### Scene 6: Docker (45 seconds)
**Terminal:**
```cmd
type Dockerfile
type docker-compose.yml
```
**Say:** "The service is containerized with Docker. Docker Compose sets up the full stack..."

#### Scene 7: CI/CD (1 minute)
**Terminal:**
```cmd
type .github\workflows\ci.yml
type .github\workflows\cd.yml
```
**Say:** "The CI pipeline runs tests and builds Docker images. The CD pipeline deploys to Kubernetes with automatic rollback..."

#### Scene 8: Kubernetes & Monitoring (30 seconds)
**Terminal:**
```cmd
type deployment\k8s\deployment.yaml
curl http://localhost:8000/metrics
curl http://localhost:8000/stats
```
**Say:** "Kubernetes manifests include health checks and autoscaling. The service exposes Prometheus metrics..."

#### Scene 9: Conclusion (30 seconds)
**Say:** "This covers our complete MLOps pipeline: data versioning with DVC, experiment tracking with MLflow, containerization, CI/CD, and monitoring. Thank you!"

---

## Phase 4: Quality Check (After Recording)

### Review Your Video
- [ ] Audio is clear and audible
- [ ] Video is smooth (no lag)
- [ ] Text is readable (terminals and browser)
- [ ] No personal information visible
- [ ] All commands executed successfully
- [ ] Total length is 4-6 minutes
- [ ] Transitions are smooth

### Edit if Needed
- [ ] Cut out long waits (pip installs, training)
- [ ] Add title slide at beginning
- [ ] Add summary slide at end
- [ ] Trim any mistakes or stutters
- [ ] Add background music (optional)

### Export Settings
- [ ] Format: MP4
- [ ] Resolution: 1920x1080 (1080p)
- [ ] Frame rate: 30 fps
- [ ] Bitrate: 5-8 Mbps
- [ ] File size: <100MB (or use cloud link)

---

## Emergency Commands (If Something Goes Wrong)

### If MLflow UI won't start:
```cmd
taskkill /F /IM mlflow.exe
mlflow ui --port 5000
```

### If API won't start (port in use):
```cmd
set PORT=8080
python src\inference_service.py
```

### If model not found:
```cmd
python src\train.py --num_epochs 3 --batch_size 16
```

### If tests fail:
```cmd
pip install -r requirements.txt
pytest tests\ -v
```

### If imports fail:
```cmd
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick Reference Commands

### Before Recording:
```cmd
cd c:\Assignment
venv\Scripts\activate
```

### During Recording:
```cmd
# Scene 2: Start MLflow UI
mlflow ui --port 5000

# Scene 4: Start API
python src\inference_service.py

# Scene 4: Test API (new terminal)
python scripts\test_api.py

# Scene 5: Run tests
pytest tests\ -v

# Scene 8: Show metrics
curl http://localhost:8000/metrics
curl http://localhost:8000/stats
```

---

## Time Budget

- Introduction: 30s
- MLflow: 60s
- Code walkthrough: 45s
- API demo: 75s
- Testing: 30s
- Docker: 45s
- CI/CD: 60s
- K8s & Monitoring: 30s
- Conclusion: 30s
**Total: ~5 minutes**

---

## Final Checklist Before You Start Recording

- [ ] Model trained successfully
- [ ] MLflow UI accessible
- [ ] Sample data generated
- [ ] All tests passing
- [ ] Virtual environment activated
- [ ] Terminal font increased
- [ ] Browser tabs prepared
- [ ] Screen recorder ready
- [ ] Microphone working
- [ ] Quiet environment
- [ ] Script reviewed
- [ ] Practice run completed

---

## After Recording

1. Save raw recording
2. Watch it once
3. Note any issues
4. Edit if needed
5. Export final version
6. Check file size
7. Upload or share link
8. Add to submission

---

## Support

If you have issues:
1. Check VIDEO_RECORDING_GUIDE.md
2. Run TEST_BEFORE_VIDEO.bat
3. Review error messages
4. Check README.md troubleshooting section

---

**You're ready to record! Good luck! 🎬🚀**

Remember: It doesn't have to be perfect. Show the working pipeline, explain briefly, and demonstrate the key features. The code quality and completeness speak for themselves!
