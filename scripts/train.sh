#!/bin/bash

# Training script for Cats vs Dogs classifier
# Usage: ./scripts/train.sh

set -e

echo "==================================="
echo "Training Cats vs Dogs Classifier"
echo "==================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Default parameters
DATA_DIR=${DATA_DIR:-"data/raw"}
MODEL_NAME=${MODEL_NAME:-"baseline"}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_EPOCHS=${NUM_EPOCHS:-10}
LEARNING_RATE=${LEARNING_RATE:-0.001}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"cats_vs_dogs"}

echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Model: $MODEL_NAME"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR does not exist"
    echo "Please download the dataset and place it in $DATA_DIR"
    exit 1
fi

# Start MLflow UI in background
echo "Starting MLflow UI on http://localhost:5000"
mlflow ui --host 0.0.0.0 --port 5000 &
MLFLOW_PID=$!

# Sleep to let MLflow UI start
sleep 3

# Train model
echo "Starting training..."
python src/train.py \
    --data_dir "$DATA_DIR" \
    --model_name "$MODEL_NAME" \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --experiment_name "$EXPERIMENT_NAME"

echo ""
echo "Training completed!"
echo "Model saved to: models/best_model.pth"
echo "MLflow UI: http://localhost:5000"
echo ""
echo "To stop MLflow UI: kill $MLFLOW_PID"
