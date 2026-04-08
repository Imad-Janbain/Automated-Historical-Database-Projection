.PHONY: install dev demo train tune grid progressive analyze methodology-diagram test lint format docker-build docker-run clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

grid:
	python scripts/run_grid.py --grid configs/seine_grid.yaml

progressive:
	python scripts/run_progressive.py --db outputs/seine_grid/results.csv --series data/seine_water_levels.csv --look-back 48 --available Water_level_Honfleur Water_level_Tancarville Water_level_Caudebec Water_level_Duclair Water_level_Rouen --targets Conductivity_Tancarville_Surface Dissolved_Oxygen_Tancarville_Surface Turbidity_Tancarville_Bottom --output outputs/seine_grid/historical_reconstruction.csv

methodology-diagram:
	python scripts/make_methodology_diagram.py

analyze:
	python scripts/analyze.py --config configs/synthetic_demo.yaml

demo:
	python scripts/generate_synthetic_demo.py --out data/synthetic.csv
	python scripts/train.py --config configs/synthetic_demo.yaml

train:
	python scripts/train.py --config configs/seine_water_quality.yaml

tune:
	python scripts/tune.py --config configs/seine_water_quality.yaml

test:
	pytest

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

docker-build:
	docker build -t deep-ts-imputer:latest -f docker/Dockerfile .

docker-run:
	docker compose -f docker/docker-compose.yml run --rm trainer

clean:
	rm -rf outputs/ .pytest_cache/ .ruff_cache/ .mypy_cache/ build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
