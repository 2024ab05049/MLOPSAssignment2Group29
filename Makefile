.PHONY: help setup install train test test-unit test-smoke serve build run deploy clean

help:
	@echo "MLOps Pipeline - Cats vs Dogs Classifier"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup          - Setup environment and install dependencies"
	@echo "  make install        - Install Python dependencies"
	@echo "  make data           - Generate sample dataset"
	@echo "  make train          - Train the model"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-smoke     - Run smoke tests only"
	@echo "  make serve          - Start API server"
	@echo "  make test-api       - Test API endpoints"
	@echo "  make build          - Build Docker image"
	@echo "  make run            - Run Docker container"
	@echo "  make deploy         - Deploy with docker-compose"
	@echo "  make deploy-k8s     - Deploy to Kubernetes"
	@echo "  make clean          - Clean generated files"
	@echo "  make mlflow         - Start MLflow UI"

setup:
	@echo "Setting up environment..."
	python -m venv venv || python3 -m venv venv
	@echo "Activate virtual environment with:"
	@echo "  source venv/bin/activate  (Linux/Mac)"
	@echo "  venv\\Scripts\\activate    (Windows)"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

data:
	python scripts/generate_sample_data.py --num_samples 100

train:
	python src/train.py \
		--data_dir data/raw \
		--model_name baseline \
		--num_epochs 10 \
		--batch_size 32 \
		--learning_rate 0.001

test: test-unit

test-unit:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-smoke:
	python deployment/smoke_tests.py

serve:
	python src/inference_service.py

test-api:
	python scripts/test_api.py

build:
	docker build -t catsdogs-api:latest .

run:
	docker run -p 8000:8000 -v $(PWD)/models:/app/models catsdogs-api:latest

deploy:
	docker-compose up -d

deploy-down:
	docker-compose down

deploy-k8s:
	kubectl apply -f deployment/k8s/

undeploy-k8s:
	kubectl delete -f deployment/k8s/

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true
	rm -rf artifacts/*.png 2>/dev/null || true
	@echo "Cleaned generated files"

lint:
	flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503

format:
	black src/ tests/ --line-length=120

git-setup:
	git init
	dvc init
	git add .gitignore .dvc/config .dvc/.dvcignore
	git commit -m "Initial commit: MLOps pipeline setup"

all: setup install data train test build

# Development workflow
dev-setup: setup install data
	@echo ""
	@echo "Development environment ready!"
	@echo "Next steps:"
	@echo "  1. Activate venv: source venv/bin/activate"
	@echo "  2. Train model: make train"
	@echo "  3. Test API: make serve (in one terminal) && make test-api (in another)"

# Quick test workflow
quick-test: data train serve

# Production deployment workflow
prod-deploy: test build deploy

# Show status
status:
	@echo "=== Git Status ==="
	@git status --short || echo "Not a git repository"
	@echo ""
	@echo "=== DVC Status ==="
	@dvc status || echo "DVC not initialized"
	@echo ""
	@echo "=== Docker Images ==="
	@docker images | grep catsdogs || echo "No catsdogs images found"
	@echo ""
	@echo "=== Docker Containers ==="
	@docker ps -a | grep catsdogs || echo "No catsdogs containers found"
	@echo ""
	@echo "=== Kubernetes Pods ==="
	@kubectl get pods -l app=catsdogs-api 2>/dev/null || echo "No Kubernetes pods found"
