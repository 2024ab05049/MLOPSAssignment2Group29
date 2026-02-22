@echo off
REM Setup script for the MLOps pipeline project (Windows)
REM Usage: scripts\setup.bat

echo ==========================================
echo MLOps Pipeline Setup
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo X Python not found. Please install Python 3.10 or higher.
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo √ Python %python_version% found
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist venv (
    echo   Virtual environment already exists
) else (
    python -m venv venv
    echo √ Virtual environment created
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo √ Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip -q
echo √ pip upgraded
echo.

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt -q
echo √ Dependencies installed
echo.

REM Create necessary directories
echo Creating project directories...
if not exist data\raw\cat mkdir data\raw\cat
if not exist data\raw\dog mkdir data\raw\dog
if not exist data\processed mkdir data\processed
if not exist models mkdir models
if not exist logs mkdir logs
if not exist artifacts mkdir artifacts
echo √ Directories created
echo.

REM Initialize DVC
echo Initializing DVC...
if exist .dvc (
    echo   DVC already initialized
) else (
    dvc init
    echo √ DVC initialized
)
echo.

REM Initialize Git
echo Checking Git repository...
if exist .git (
    echo   Git repository already exists
) else (
    git init
    echo √ Git repository initialized
)
echo.

REM Create .env file
echo Creating .env file...
if exist .env (
    echo   .env file already exists
) else (
    (
        echo # Environment variables for MLOps pipeline
        echo MODEL_PATH=models/best_model.pth
        echo PORT=8000
        echo API_URL=http://localhost:8000
    ) > .env
    echo √ .env file created
)
echo.

REM Check Docker
echo Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ! Docker not found. Install Docker to use containerization features.
) else (
    for /f "tokens=3" %%i in ('docker --version') do set docker_version=%%i
    echo √ Docker found
)
echo.

REM Check Kubernetes
echo Checking Kubernetes...
kubectl version --client >nul 2>&1
if errorlevel 1 (
    echo ! kubectl not found. Install kubectl to use Kubernetes deployment.
) else (
    echo √ kubectl found
)
echo.

REM Summary
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo   1. Download the Cats vs Dogs dataset from Kaggle
echo   2. Place images in data\raw\cat\ and data\raw\dog\
echo   3. Activate virtual environment: venv\Scripts\activate.bat
echo   4. Train the model: scripts\train.bat
echo   5. Test the API: python src\inference_service.py
echo.
echo For more information, see README.md
echo.
pause
