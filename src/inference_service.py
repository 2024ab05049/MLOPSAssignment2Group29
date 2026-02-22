"""
FastAPI inference service for Cats vs Dogs classification.
Provides REST API endpoints for model predictions with monitoring.
"""

import os
import io
import time
import json
import logging
from typing import Dict, Any
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from model import get_model
from data_preprocessing import get_val_transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="Binary image classification API for pet adoption platform",
    version="1.0.0"
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'inference_requests_total',
    'Total number of inference requests',
    ['endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'inference_request_latency_seconds',
    'Inference request latency',
    ['endpoint']
)
PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total number of predictions by class',
    ['predicted_class']
)

# Global model variable
model = None
device = None
transform = None
class_names = ["cat", "dog"]
model_metadata = {}

# Request/Response logging storage (in-memory for demo)
request_logs = []
MAX_LOGS = 1000


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str
    version: str


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_label: int
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float
    timestamp: str


def load_model(model_path: str = "models/best_model.pth"):
    """Load the trained model."""
    global model, device, transform, model_metadata

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading model from {model_path} on device: {device}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Initialize model
        model = get_model("baseline", num_classes=2, dropout_rate=0.5)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        # Load transforms
        transform = get_val_transforms()

        # Load metadata if available
        metadata_path = "models/model_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)

        logger.info("Model loaded successfully")
        logger.info(f"Model metadata: {model_metadata}")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        model_path = os.getenv("MODEL_PATH", "models/best_model.pth")
        load_model(model_path)
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
        # Continue without model - health check will show model not loaded


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Cats vs Dogs Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns service status and model readiness.
    """
    start_time = time.time()

    response = HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )

    latency = (time.time() - start_time) * 1000
    logger.info(f"Health check completed in {latency:.2f}ms")

    REQUEST_COUNT.labels(endpoint='health', status='success').inc()

    return response


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Prediction endpoint.
    Accepts an image file and returns the predicted class with confidence scores.

    Args:
        file: Image file (jpg, jpeg, png)

    Returns:
        Prediction results with class label, confidence, and probabilities
    """
    request_start_time = time.time()

    # Check if model is loaded
    if model is None:
        REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Transform image
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Inference
        inference_start = time.time()
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        inference_time = (time.time() - inference_start) * 1000

        # Prepare response
        pred_label = predicted.item()
        pred_class = class_names[pred_label]
        conf_score = confidence.item()

        probs_dict = {
            class_names[i]: float(probabilities[0][i].item())
            for i in range(len(class_names))
        }

        response = PredictionResponse(
            predicted_class=pred_class,
            predicted_label=pred_label,
            confidence=conf_score,
            probabilities=probs_dict,
            inference_time_ms=inference_time,
            timestamp=datetime.utcnow().isoformat()
        )

        # Log request
        log_entry = {
            "timestamp": response.timestamp,
            "filename": file.filename,
            "predicted_class": pred_class,
            "confidence": conf_score,
            "inference_time_ms": inference_time
        }
        request_logs.append(log_entry)
        if len(request_logs) > MAX_LOGS:
            request_logs.pop(0)

        # Update metrics
        total_latency = (time.time() - request_start_time) * 1000
        REQUEST_COUNT.labels(endpoint='predict', status='success').inc()
        REQUEST_LATENCY.labels(endpoint='predict').observe(total_latency / 1000)
        PREDICTION_COUNT.labels(predicted_class=pred_class).inc()

        logger.info(f"Prediction: {pred_class} ({conf_score:.4f}) in {inference_time:.2f}ms")

        return response

    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    Returns metrics in Prometheus format.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/logs")
async def get_logs(limit: int = 100):
    """
    Get recent prediction logs.

    Args:
        limit: Number of recent logs to return (default: 100)

    Returns:
        List of recent prediction logs
    """
    return {
        "total_requests": len(request_logs),
        "logs": request_logs[-limit:]
    }


@app.get("/stats")
async def get_stats():
    """
    Get service statistics.

    Returns:
        Service statistics including request counts and predictions
    """
    if not request_logs:
        return {
            "total_requests": 0,
            "predictions_by_class": {},
            "average_inference_time_ms": 0
        }

    # Calculate statistics
    cat_count = sum(1 for log in request_logs if log['predicted_class'] == 'cat')
    dog_count = sum(1 for log in request_logs if log['predicted_class'] == 'dog')
    avg_inference_time = np.mean([log['inference_time_ms'] for log in request_logs])

    return {
        "total_requests": len(request_logs),
        "predictions_by_class": {
            "cat": cat_count,
            "dog": dog_count
        },
        "average_inference_time_ms": float(avg_inference_time),
        "average_confidence": float(np.mean([log['confidence'] for log in request_logs]))
    }


@app.get("/model-info")
async def model_info():
    """
    Get model information and metadata.

    Returns:
        Model metadata and configuration
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_loaded": True,
        "device": str(device),
        "class_names": class_names,
        "metadata": model_metadata
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
