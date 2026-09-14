.PHONY: install train api test docker-build docker-run clean

install:
	python -m pip install -r requirements.txt

train:
	python -m src.train
	python -m src.evaluate

api:
	uvicorn src.app:app --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

docker-build:
	docker compose build

docker-run:
	docker compose up

clean:
	python -c "from pathlib import Path; [p.unlink() for p in Path('outputs').glob('**/*.log')]"
