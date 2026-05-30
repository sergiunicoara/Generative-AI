# GraphRAG — common developer workflows
# Usage: make <target>
# Requires: Python 3.11+, Docker (for services)

PYTHON    ?= python
PYTEST    ?= $(PYTHON) -m pytest
UVICORN   ?= $(PYTHON) -m uvicorn
PIP       ?= $(PYTHON) -m pip

.PHONY: help install install-dev lock test test-integration test-load test-all lint \
        api dashboard backup services-up services-down \
        community-rebuild re-embed entity-migrate

# ── Meta ───────────────────────────────────────────────────────────────────────

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

# ── Install ────────────────────────────────────────────────────────────────────

install:       ## Install all Python dependencies
	$(PIP) install -r requirements.txt

install-dev:   ## Install + development extras (pytest, ruff, pip-tools)
	$(PIP) install -r requirements.txt ruff pip-tools

lock:          ## Regenerate requirements.lock from requirements.txt (reproducible builds)
	pip-compile requirements.txt --output-file requirements.lock --strip-extras

# ── Testing ────────────────────────────────────────────────────────────────────

test:          ## Run unit tests only (fast, no live services)
	$(PYTEST) tests/unit/ -x

test-integration: ## Run integration tests (AsyncMock, no live services)
	$(PYTEST) tests/integration/ -v

test-load:     ## Run concurrency load tests
	$(PYTEST) tests/load/ -v

test-all:      ## Run the complete test suite
	$(PYTEST) tests/ -v

test-safety:   ## Run safety-path integration tests
	$(PYTEST) tests/integration/test_safety_paths.py -v

test-ops:      ## Run operational-path integration tests
	$(PYTEST) tests/integration/test_operational_paths.py -v

# ── Lint ───────────────────────────────────────────────────────────────────────

lint:          ## Lint with ruff (fast, replaces flake8 + isort)
	$(PYTHON) -m ruff check graphrag/ api/ scripts/ tests/

lint-fix:      ## Auto-fix lint issues
	$(PYTHON) -m ruff check --fix graphrag/ api/ scripts/ tests/

# ── Services ───────────────────────────────────────────────────────────────────

services-up:   ## Start Neo4j + Redis via Docker Compose
	docker compose up -d neo4j redis

services-down: ## Stop all Docker Compose services
	docker compose down

# ── API & Dashboard ────────────────────────────────────────────────────────────

api:           ## Start the FastAPI server (hot-reload, dev mode)
	$(UVICORN) api.main:app --reload --port 8000

dashboard:     ## Start the Dash admin dashboard in standalone mode (dev)
	$(PYTHON) -m graphrag.dashboard.app

# ── Maintenance scripts ────────────────────────────────────────────────────────

community-rebuild:  ## Rebuild stale communities for a tenant (TENANT= required)
	$(PYTHON) scripts/community_rebuild.py --tenant $(TENANT)

re-embed:      ## Re-embed entities with a new model (TENANT= MODEL= required)
	$(PYTHON) scripts/re_embed.py --tenant $(TENANT) --model $(MODEL)

entity-migrate: ## Migrate entity type (OLD_TYPE= NEW_TYPE= TENANT= required)
	$(PYTHON) scripts/entity_type_migration.py \
	  --old-type $(OLD_TYPE) --new-type $(NEW_TYPE) --tenant $(TENANT)

backup:        ## Export full graph to NDJSON (OUTPUT= optional, default ./backup)
	$(PYTHON) scripts/kg_backup.py --output $(or $(OUTPUT),./backup)

backup-s3:     ## Export graph to S3 (S3_BUCKET= required)
	$(PYTHON) scripts/kg_backup.py --s3-bucket $(S3_BUCKET)
