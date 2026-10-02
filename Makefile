.PHONY: setup data lint test clean

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

data:
	python -m src.data.load FD001

lint:
	ruff check .

test:
	pytest -q

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
