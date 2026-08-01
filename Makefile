.PHONY: setup lint format test train dashboard docker-build docker-run clean

PYTHON := python
PIP := pip
VENV := .venv

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	$(PIP) install -e .

lint:
	ruff check .
	mypy src/
	black --check .

format:
	black .
	ruff check --fix .

test:
	pytest --cov=src/demandiq --cov-report=term --cov-report=xml --cov-fail-under=85

train:
	$(PYTHON) -m demandiq.data.generate_synthetic --out data/raw/orders.csv --seed 42
	$(PYTHON) -m demandiq.pipeline

dashboard:
	streamlit run src/demandiq/dashboard/app.py --server.port=8501 --server.address=0.0.0.0

docker-build:
	docker build -t demandiq:latest .

docker-run:
	docker run -p 8501:8501 -v $(PWD)/models:/app/models demandiq:latest

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .mypy_cache/ .coverage coverage.xml htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
