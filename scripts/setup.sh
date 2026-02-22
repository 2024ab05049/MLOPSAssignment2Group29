#!/bin/bash

# Setup script for the MLOps pipeline project
# Usage: ./scripts/setup.sh

set -e

echo "=========================================="
echo "MLOps Pipeline Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "✗ Python 3.10 or higher is required (found $python_version)"
    exit 1
fi
echo "✓ Python $python_version found"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "  Virtual environment already exists"
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q
echo "✓ pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt -q
echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/raw/cat data/raw/dog
mkdir -p data/processed
mkdir -p models
mkdir -p logs
mkdir -p artifacts
echo "✓ Directories created"

# Initialize DVC
echo ""
echo "Initializing DVC..."
if [ -d ".dvc" ]; then
    echo "  DVC already initialized"
else
    dvc init
    echo "✓ DVC initialized"
fi

# Initialize Git (if not already initialized)
echo ""
echo "Checking Git repository..."
if [ -d ".git" ]; then
    echo "  Git repository already exists"
else
    git init
    echo "✓ Git repository initialized"
fi

# Create .env file if it doesn't exist
echo ""
echo "Creating .env file..."
if [ -f ".env" ]; then
    echo "  .env file already exists"
else
    cat > .env << EOF
# Environment variables for MLOps pipeline
MODEL_PATH=models/best_model.pth
PORT=8000
API_URL=http://localhost:8000
EOF
    echo "✓ .env file created"
fi

# Check Docker installation
echo ""
echo "Checking Docker..."
if command -v docker &> /dev/null; then
    docker_version=$(docker --version | awk '{print $3}' | tr -d ',')
    echo "✓ Docker $docker_version found"
else
    echo "⚠ Docker not found. Install Docker to use containerization features."
fi

# Check Kubernetes
echo ""
echo "Checking Kubernetes..."
if command -v kubectl &> /dev/null; then
    kubectl_version=$(kubectl version --client --short 2>/dev/null | awk '{print $3}')
    echo "✓ kubectl $kubectl_version found"
else
    echo "⚠ kubectl not found. Install kubectl to use Kubernetes deployment."
fi

# Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Download the Cats vs Dogs dataset from Kaggle"
echo "  2. Place images in data/raw/cat/ and data/raw/dog/"
echo "  3. Activate virtual environment: source venv/bin/activate"
echo "  4. Train the model: ./scripts/train.sh"
echo "  5. Test the API: python src/inference_service.py"
echo ""
echo "For more information, see README.md"
echo ""
