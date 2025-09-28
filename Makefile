# Makefile for Review Search Copilot

.PHONY: help install test build run clean deploy health

# Default target
help:
	@echo "Review Search Copilot - Available commands:"
	@echo ""
	@echo "Development:"
	@echo "  install     Install dependencies"
	@echo "  test        Run tests"
	@echo "  run         Run application locally"
	@echo "  health      Run health checks"
	@echo ""
	@echo "Docker:"
	@echo "  build       Build Docker image"
	@echo "  docker-run  Run with Docker"
	@echo "  docker-dev  Run development setup with Docker Compose"
	@echo ""
	@echo "Production:"
	@echo "  deploy-prod Deploy production setup"
	@echo "  clean       Clean up build artifacts"

# Development commands
install:
	pip install -r requirements.txt

test:
	python run_tests.py

run:
	./start.sh

health:
	python health_check.py

# Docker commands
build:
	docker build -t review-search-app .

docker-run: build
	docker run -p 8501:8501 \
		-v $(PWD)/data:/app/data:ro \
		-v $(PWD)/logs:/app/logs \
		-e ENVIRONMENT=development \
		review-search-app

docker-dev:
	docker-compose up --build

# Production commands
deploy-prod:
	cp .env.production .env
	docker-compose --profile production up -d --build

# Utility commands
clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	docker system prune -f

# Data pipeline (if preprocessing scripts exist)
data-prep:
	python nlp/10_product_prep.py
	python nlp/11_build_product_embeddings.py
	python nlp/12_product_prep.py

# Validation
validate:
	python -c "from config import config; config.validate(); print('✅ Configuration valid')"
	python health_check.py
	python run_tests.py

# Quick deployment to common platforms
deploy-hf:
	@echo "Manual steps for Hugging Face Spaces:"
	@echo "1. Create new Space on huggingface.co"
	@echo "2. Upload all files except .git/, __pycache__/, logs/"
	@echo "3. Set environment variables in Space settings"
	@echo "4. Space will auto-deploy using Dockerfile"

deploy-render:
	@echo "Manual steps for Render:"
	@echo "1. Connect GitHub repo to Render"
	@echo "2. Choose 'Web Service' with Docker"
	@echo "3. Set environment variables in dashboard"
	@echo "4. Deploy will start automatically"

# Development workflow
dev-setup: install validate
	@echo "✅ Development setup complete"
	@echo "Run 'make run' to start the application"

# Production workflow  
prod-setup: build validate
	@echo "✅ Production setup complete"
	@echo "Run 'make deploy-prod' to start production deployment"