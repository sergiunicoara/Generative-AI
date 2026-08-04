.PHONY: up down test test-unit test-integration demo lint format

up:
	docker compose up -d neo4j

down:
	docker compose down

test-unit:
	pytest tests/unit

test-integration: up
	pytest tests/integration tests/eval tests/security

test: up
	pytest tests/

demo: up
	python demo_volkswagen.py

lint:
	ruff check src api tests

format:
	ruff format src api tests
