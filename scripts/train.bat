@echo off
REM Training script for Cats vs Dogs classifier (Windows)
REM Usage: scripts\train.bat

echo ===================================
echo Training Cats vs Dogs Classifier
echo ===================================
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Default parameters
if not defined DATA_DIR set DATA_DIR=data\raw
if not defined MODEL_NAME set MODEL_NAME=baseline
if not defined BATCH_SIZE set BATCH_SIZE=32
if not defined NUM_EPOCHS set NUM_EPOCHS=10
if not defined LEARNING_RATE set LEARNING_RATE=0.001
if not defined EXPERIMENT_NAME set EXPERIMENT_NAME=cats_vs_dogs

echo Configuration:
echo   Data directory: %DATA_DIR%
echo   Model: %MODEL_NAME%
echo   Batch size: %BATCH_SIZE%
echo   Epochs: %NUM_EPOCHS%
echo   Learning rate: %LEARNING_RATE%
echo.

REM Check if data directory exists
if not exist "%DATA_DIR%" (
    echo Error: Data directory %DATA_DIR% does not exist
    echo Please download the dataset and place it in %DATA_DIR%
    exit /b 1
)

REM Start MLflow UI in background
echo Starting MLflow UI on http://localhost:5000
start /B mlflow ui --host 0.0.0.0 --port 5000

REM Sleep to let MLflow UI start
timeout /t 3 /nobreak >nul

REM Train model
echo Starting training...
python src\train.py ^
    --data_dir "%DATA_DIR%" ^
    --model_name "%MODEL_NAME%" ^
    --batch_size %BATCH_SIZE% ^
    --num_epochs %NUM_EPOCHS% ^
    --learning_rate %LEARNING_RATE% ^
    --experiment_name "%EXPERIMENT_NAME%"

echo.
echo Training completed!
echo Model saved to: models\best_model.pth
echo MLflow UI: http://localhost:5000
echo.
pause
