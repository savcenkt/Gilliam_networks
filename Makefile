# Makefile for Gilliam Networks Pipeline
# Automates common tasks for the semantic network analysis project

.PHONY: help setup test clean run-all run-prepare run-embed run-network run-analyze explore lint format check-deps

# Default target - show help
help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║     Gilliam Networks - Semantic Network Analysis Pipeline    ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Available commands:"
	@echo ""
	@echo "  Setup & Dependencies:"
	@echo "    make setup        - Install all dependencies"
	@echo "    make check-deps   - Check if dependencies are installed"
	@echo ""
	@echo "  Pipeline Execution:"
	@echo "    make run-all      - Run complete pipeline (all phases)"
	@echo "    make run-prepare  - Run data preparation phase"
	@echo "    make run-embed    - Generate BERT embeddings"
	@echo "    make run-network  - Construct semantic networks"
	@echo "    make run-analyze  - Perform analysis and generate reports"
	@echo ""
	@echo "  Interactive & Testing:"
	@echo "    make explore      - Launch interactive Marimo notebook"
	@echo "    make test         - Run test suite"
	@echo "    make test-cov     - Run tests with coverage report"
	@echo ""
	@echo "  Code Quality:"
	@echo "    make lint         - Run code linting (flake8)"
	@echo "    make format       - Format code with black"
	@echo "    make type-check   - Run type checking with mypy"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make clean        - Clean generated files and caches"
	@echo "    make clean-data   - Clean only data outputs (preserves code)"
	@echo "    make reset        - Full reset (clean + remove checkpoints)"
	@echo ""
	@echo "  Quick Commands:"
	@echo "    make quick-test   - Test on small subset of data"
	@echo "    make status       - Show pipeline progress status"
	@echo ""

# Setup and installation
setup:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully"
	@echo ""
	@echo "🔍 Verifying installation..."
	python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	python -c "import networkx; print(f'NetworkX: {networkx.__version__}')"
	@echo "✅ Setup complete!"

# Check if dependencies are installed
check-deps:
	@echo "🔍 Checking dependencies..."
	@python -c "import pkg_resources; pkg_resources.require(open('requirements.txt').read().splitlines())" && echo "✅ All dependencies satisfied" || echo "❌ Missing dependencies - run 'make setup'"

# Run complete pipeline
run-all:
	@echo "🚀 Running complete pipeline..."
	python scripts/run_pipeline.py --phase all --config config.yaml

# Run individual phases
run-prepare:
	@echo "📝 Running data preparation..."
	python scripts/run_pipeline.py --phase prepare --config config.yaml

run-embed:
	@echo "🤖 Generating embeddings..."
	python scripts/run_pipeline.py --phase embed --config config.yaml

run-network:
	@echo "🕸️ Constructing networks..."
	python scripts/run_pipeline.py --phase network --config config.yaml

run-analyze:
	@echo "📊 Running analysis..."
	python scripts/run_pipeline.py --phase analyze --config config.yaml

# Launch interactive Marimo notebook
explore:
	@echo "🎨 Launching interactive Marimo notebook..."
	@echo "Opening in browser..."
	cd src/interactive && marimo run marimo_explorer.py

# Run tests
test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v --tb=short

# Run tests with coverage
test-cov:
	@echo "🧪 Running tests with coverage..."
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "📊 Coverage report generated in htmlcov/index.html"

# Quick test on subset
quick-test:
	@echo "⚡ Running quick test on data subset..."
	python scripts/run_pipeline.py --phase embed --config config.yaml --dry-run
	@echo "✅ Quick test complete"

# Code quality
lint:
	@echo "🔍 Running linter..."
	flake8 src/ scripts/ tests/ --max-line-length=120 --ignore=E203,W503

format:
	@echo "✨ Formatting code with black..."
	black src/ scripts/ tests/ --line-length=120

type-check:
	@echo "🔍 Running type checker..."
	mypy src/ scripts/ --ignore-missing-imports

# Show pipeline status
status:
	@echo "📊 Pipeline Progress Status:"
	@python -c "import json; import os; \
		f = 'pipeline_progress.json'; \
		if os.path.exists(f): \
			d = json.load(open(f)); \
			print('Phases:'); \
			for k, v in d.get('phases', {}).items(): \
				s = '✅' if v else '⏭️'; \
				print(f'  {s} {k}'); \
		else: \
			print('No progress file found. Pipeline not started.');"

# Cleaning commands
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf __pycache__ **/__pycache__ **/**/__pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -f pipeline.log
	rm -f pipeline_progress.json
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned cache and temporary files"

clean-data:
	@echo "🧹 Cleaning data outputs..."
	rm -rf data/processed/*
	rm -rf data/results/*
	@echo "✅ Data outputs cleaned"

reset: clean clean-data
	@echo "🔄 Performing full reset..."
	rm -rf checkpoints/*
	rm -f pipeline_progress.json
	@echo "✅ Full reset complete"

# Development helpers
dev-install:
	@echo "📦 Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy jupyter notebook

# Create directories if they don't exist
init-dirs:
	@mkdir -p data/{raw,processed/{embeddings,networks},results/{figures,tables,reports}}
	@mkdir -p checkpoints
	@mkdir -p logs
	@echo "✅ Directory structure initialized"

# Help for specific phases
help-embed:
	@echo "📖 Embedding Generation Help:"
	@echo ""
	@echo "Generate BERT embeddings for child narratives:"
	@echo "  make run-embed"
	@echo ""
	@echo "Configuration (config.yaml):"
	@echo "  - model: BERT model to use (default: bert-base-uncased)"
	@echo "  - batch_size: Processing batch size (default: 8)"
	@echo "  - device: cuda/cpu/auto"

help-analyze:
	@echo "📖 Analysis Help:"
	@echo ""
	@echo "Analyze semantic networks:"
	@echo "  make run-analyze"
	@echo ""
	@echo "Generates:"
	@echo "  - Group comparison statistics"
	@echo "  - Small-world coefficients"
	@echo "  - Visualizations"
	@echo "  - Markdown reports"

# Docker support (optional)
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t gilliam-networks .

docker-run:
	@echo "🐳 Running in Docker container..."
	docker run -v $(PWD)/data:/app/data gilliam-networks make run-all

# Git helpers
git-stats:
	@echo "📊 Repository Statistics:"
	@echo "Lines of code:"
	@find src scripts tests -name "*.py" | xargs wc -l | tail -1
	@echo ""
	@echo "Number of Python files:"
	@find src scripts tests -name "*.py" | wc -l
	@echo ""
	@echo "Latest commit:"
	@git log -1 --oneline

# Variables for configuration
CONFIG ?= config.yaml
VERBOSE ?= false
FORCE ?= false

# Advanced run with options
run-custom:
	python scripts/run_pipeline.py \
		--config $(CONFIG) \
		$(if $(filter true,$(VERBOSE)),--verbose,) \
		$(if $(filter true,$(FORCE)),--force,) \
		--phase $(PHASE)

.SILENT: help status git-stats