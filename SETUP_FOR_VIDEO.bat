@echo off
echo ========================================
echo MLOps Pipeline - Quick Setup for Video
echo ========================================
echo.

REM Check if running in Assignment directory
if not exist "src\" (
    echo Error: Please run this from the Assignment directory
    pause
    exit /b 1
)

echo [1/6] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    echo Make sure Python 3.10+ is installed
    pause
    exit /b 1
)
echo     Done!
echo.

echo [2/6] Activating virtual environment...
call venv\Scripts\activate.bat
echo     Done!
echo.

echo [3/6] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo     Done!
echo.

echo [4/6] Installing dependencies (this may take 2-3 minutes)...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)
echo     Done!
echo.

echo [5/6] Creating directories...
if not exist "data\raw\cat" mkdir data\raw\cat
if not exist "data\raw\dog" mkdir data\raw\dog
if not exist "data\processed" mkdir data\processed
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "artifacts" mkdir artifacts
echo     Done!
echo.

echo [6/6] Generating sample dataset (50 cats, 50 dogs)...
python scripts\generate_sample_data.py --num_samples 50
if errorlevel 1 (
    echo Error: Failed to generate sample data
    pause
    exit /b 1
)
echo     Done!
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps for video recording:
echo.
echo 1. Train the model (5-10 minutes):
echo    python src\train.py --num_epochs 5 --batch_size 16
echo.
echo 2. Start MLflow UI:
echo    mlflow ui --port 5000
echo    Then open: http://localhost:5000
echo.
echo 3. Test the API:
echo    Terminal 1: python src\inference_service.py
echo    Terminal 2: python scripts\test_api.py
echo.
echo 4. Run tests:
echo    pytest tests\ -v
echo.
echo See VIDEO_RECORDING_GUIDE.md for detailed instructions!
echo.
pause
