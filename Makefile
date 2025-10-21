.PHONY: install test lint train

install:
	python -m pip install --upgrade pip;
	python -m pip install -r requirements.txt;
	python -m pip install -e .

test:
	python -m pytest tests -q

lint:
	# add linting commands here if desired
	@echo "No linter configured"

train:
	python main.py --train_path data/raw/train.csv --test_path data/raw/test.csv --submission_path data/raw/sample_submission.csv
.PHONY: train test lint clean

train:
	python main.py

test:
	pytest tests/ -v

lint:
	black src/ tests/ main.py
	flake8 src/ tests/ main.py

clean:
	rm -rf models/*
	rm -rf data/processed/*
	rm -rf mlruns/*
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

install:
	pip install -e .

experiment:
	mlflow ui