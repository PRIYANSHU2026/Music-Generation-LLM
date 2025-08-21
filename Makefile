# Music Generation LLM Makefile
# Common commands for development and deployment

.PHONY: help install test lint format clean docker-build docker-run docker-stop docker-clean serve dev

# Default target
help:
	@echo "Music Generation LLM - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install     Install dependencies"
	@echo "  dev         Start development server with auto-reload"
	@echo "  serve       Start production server"
	@echo "  test        Run tests"
	@echo "  test-cov    Run tests with coverage"
	@echo "  lint        Run linting checks"
	@echo "  format      Format code with Black and isort"
	@echo "  type-check  Run type checking with mypy"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    Build Docker image"
	@echo "  docker-run      Run Docker container"
	@echo "  docker-stop     Stop Docker container"
	@echo "  docker-clean    Clean Docker containers and images"
	@echo "  docker-compose  Start with Docker Compose"
	@echo "  docker-prod     Start production stack with Docker Compose"
	@echo ""
	@echo "Utilities:"
	@echo "  clean       Clean Python cache and build files"
	@echo "  db-init     Initialize database"
	@echo "  cli-help    Show CLI help"

# Development commands
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -r requirements.txt[dev]

dev:
	@echo "Starting development server..."
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

serve:
	@echo "Starting production server..."
	uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	@echo "Running tests..."
	pytest

test-cov:
	@echo "Running tests with coverage..."
	pytest --cov=app --cov=cli --cov-report=html --cov-report=term

lint:
	@echo "Running linting checks..."
	flake8 app/ cli/ tests/
	mypy app/ cli/

format:
	@echo "Formatting code..."
	black app/ cli/ tests/
	isort app/ cli/ tests/

type-check:
	@echo "Running type checks..."
	mypy app/ cli/

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -t music-generation-llm .

docker-run:
	@echo "Running Docker container..."
	docker run -d -p 8000:8000 --name music-llm-container music-generation-llm

docker-stop:
	@echo "Stopping Docker container..."
	docker stop music-llm-container || true
	docker rm music-llm-container || true

docker-clean:
	@echo "Cleaning Docker containers and images..."
	docker stop music-llm-container || true
	docker rm music-llm-container || true
	docker rmi music-generation-llm || true

docker-compose:
	@echo "Starting with Docker Compose..."
	docker-compose up -d

docker-prod:
	@echo "Starting production stack with Docker Compose..."
	docker-compose --profile production up -d

# Utility commands
clean:
	@echo "Cleaning Python cache and build files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type f -name "*.db" -delete
	rm -rf build/ dist/ .eggs/

db-init:
	@echo "Initializing database..."
	python -c "from app.db.database import init_db; init_db()"

cli-help:
	@echo "Showing CLI help..."
	music-llm --help

# Development workflow
dev-setup: install format lint type-check
	@echo "Development environment setup complete!"

# Pre-commit setup
pre-commit-setup:
	@echo "Setting up pre-commit hooks..."
	pip install pre-commit
	pre-commit install

# Production deployment
deploy: docker-build docker-run
	@echo "Deployment complete!"

# Monitoring
monitor:
	@echo "Starting monitoring stack..."
	docker-compose --profile production up -d prometheus grafana
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000 (admin/admin)"

# Backup and restore
backup:
	@echo "Creating backup..."
	tar -czf backup-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		--exclude=venv \
		--exclude=__pycache__ \
		--exclude=*.pyc \
		--exclude=.git \
		.

# Database operations
db-migrate:
	@echo "Running database migrations..."
	alembic upgrade head

db-rollback:
	@echo "Rolling back database..."
	alembic downgrade -1

# Security checks
security-check:
	@echo "Running security checks..."
	bandit -r app/ cli/
	safety check

# Performance testing
perf-test:
	@echo "Running performance tests..."
	locust -f tests/performance/locustfile.py --host=http://localhost:8000

# Documentation
docs:
	@echo "Generating documentation..."
	pdoc --html app/ --output-dir docs/
	@echo "Documentation generated in docs/ directory"

# Environment setup
env-setup:
	@echo "Setting up environment..."
	@if [ ! -f .env ]; then \
		echo "Creating .env file from template..."; \
		cp .env.example .env; \
		echo "Please edit .env file with your configuration"; \
	else \
		echo ".env file already exists"; \
	fi

# Quick start for new developers
quickstart: env-setup install dev-setup
	@echo "Quick start complete!"
	@echo "Next steps:"
	@echo "1. Edit .env file with your OpenAI API key"
	@echo "2. Run 'make dev' to start the development server"
	@echo "3. Visit http://localhost:8000/docs for API documentation"
