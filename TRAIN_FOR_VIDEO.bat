@echo off
REM Quick training script optimized for video demo

echo ========================================
echo Training Model for Video Demo
echo ========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if data exists
if not exist "data\raw\cat" (
    echo Generating sample data first...
    python scripts\generate_sample_data.py --num_samples 50
    echo.
)

echo Training parameters:
echo   - Epochs: 5 (quick demo)
echo   - Batch size: 16
echo   - Model: baseline CNN
echo.
echo This will take approximately 5-10 minutes...
echo.

REM Start MLflow UI in background
start /B mlflow ui --host 0.0.0.0 --port 5000

echo MLflow UI starting at: http://localhost:5000
echo.

REM Train the model
python src\train.py ^
    --data_dir data\raw ^
    --model_name baseline ^
    --num_epochs 5 ^
    --batch_size 16 ^
    --learning_rate 0.001 ^
    --experiment_name cats_vs_dogs ^
    --run_name video_demo_run

if errorlevel 1 (
    echo.
    echo Training failed! Check the error message above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
echo Model saved to: models\best_model.pth
echo.
echo View results in MLflow UI:
echo   http://localhost:5000
echo.
echo Next steps:
echo   1. Open MLflow UI in browser
echo   2. Start API: python src\inference_service.py
echo   3. Test API: python scripts\test_api.py
echo.

REM Show model info
echo Model file info:
dir models\best_model.pth
echo.

pause
