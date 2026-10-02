.PHONY: setup data features baseline lstm predict lint test clean

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

data:
	python -m src.data.load FD001

features: data
	python -m src.features.build FD001

baseline: features
	python -m src.models.baseline FD001

lstm: features
	python -m src.models.lstm FD001

predict: baseline
	python -m src.predict FD001

lint:
	ruff check .

test:
	pytest -q

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
