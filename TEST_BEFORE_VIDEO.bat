@echo off
REM Quick test script to verify everything works before video recording

echo ========================================
echo Pre-Video Testing Script
echo ========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo [TEST 1] Checking Python packages...
python -c "import torch; import fastapi; import mlflow; import pytest; print('All packages OK')"
if errorlevel 1 (
    echo FAIL: Missing packages
    exit /b 1
)
echo PASS
echo.

echo [TEST 2] Checking data...
if not exist "data\raw\cat\cat_0000.jpg" (
    echo FAIL: Sample data not found
    echo Run: python scripts\generate_sample_data.py --num_samples 50
    exit /b 1
)
echo PASS: Sample data exists
echo.

echo [TEST 3] Checking model...
if not exist "models\best_model.pth" (
    echo WARNING: Model not trained yet
    echo Run: python src\train.py --num_epochs 5 --batch_size 16
    echo.
) else (
    echo PASS: Model exists
    echo.
)

echo [TEST 4] Running unit tests...
pytest tests\ -v --tb=short
if errorlevel 1 (
    echo FAIL: Some tests failed
    exit /b 1
)
echo PASS: All tests passed
echo.

echo [TEST 5] Testing data preprocessing...
python -c "from src.data_preprocessing import preprocess_image; img = preprocess_image('data/raw/cat/cat_0000.jpg'); print(f'Image shape: {img.shape}'); assert img.shape == (224, 224, 3), 'Wrong shape'"
if errorlevel 1 (
    echo FAIL: Data preprocessing error
    exit /b 1
)
echo PASS
echo.

echo [TEST 6] Testing model loading...
python -c "from src.model import get_model; model = get_model('baseline'); print('Model loaded successfully')"
if errorlevel 1 (
    echo FAIL: Model loading error
    exit /b 1
)
echo PASS
echo.

echo ========================================
echo Pre-Video Testing Complete!
echo ========================================
echo.

if not exist "models\best_model.pth" (
    echo IMPORTANT: Train the model before recording:
    echo   python src\train.py --num_epochs 5 --batch_size 16
    echo.
) else (
    echo All systems ready for video recording!
    echo.
    echo Commands for your video:
    echo   1. MLflow UI:  mlflow ui --port 5000
    echo   2. Start API:  python src\inference_service.py
    echo   3. Test API:   python scripts\test_api.py
    echo   4. Run tests:  pytest tests\ -v
    echo.
)

pause
