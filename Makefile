# GraphRAG — common developer workflows
# Usage: make <target>
# Requires: Python 3.11+, Docker (for services)

PYTHON    ?= python
PYTEST    ?= $(PYTHON) scripts/run_pytest.py
UVICORN   ?= $(PYTHON) -m uvicorn
PIP       ?= $(PYTHON) -m pip

.PHONY: help install install-dev lock test test-collect test-integration test-load test-all lint \
	terraform-fmt terraform-validate terraform-test terraform-security \
        api dashboard backup services-up services-down \
        community-rebuild re-embed entity-migrate smoke-test

# ── Meta ───────────────────────────────────────────────────────────────────────

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

# ── Install ────────────────────────────────────────────────────────────────────

install:       ## Install all Python dependencies
	$(PIP) install -r requirements.txt

install-dev:   ## Install + development extras (pytest, pytest-asyncio, ruff, pip-tools)
	$(PIP) install -r requirements-dev.txt

lock:          ## Regenerate requirements.lock from requirements.txt (reproducible builds)
	pip-compile requirements.txt --output-file requirements.lock --strip-extras

# ── Testing ────────────────────────────────────────────────────────────────────

test:          ## Run unit tests only (fast, no live services)
	$(PYTEST) tests/unit/ -x

test-collect:   ## Collect every test with the deterministic plugin set
	$(PYTEST) --collect-only -q tests/

terraform-fmt:  ## Check Terraform formatting
	terraform -chdir=infra/terraform fmt -check -recursive

terraform-validate:  ## Validate Terraform without a backend
	terraform -chdir=infra/terraform init -backend=false
	terraform -chdir=infra/terraform validate -no-color

terraform-test:  ## Run provider-mocked Terraform invariant tests
	terraform -chdir=infra/terraform test -no-color

terraform-security:  ## Run Checkov when installed (no cloud credentials required)
	checkov -d infra/terraform --framework terraform

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

test-shacl:    ## Run SHACL/ontology validation tests (fast, no live services)
	$(PYTEST) tests/unit/test_shacl_validator.py tests/unit/test_ontology_lifecycle.py tests/unit/test_ontology_registry.py tests/unit/test_export_rdf.py tests/unit/test_relational_ingestion.py -v

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

smoke-test:    ## End-to-end stack health check: unit tests + demo + API ping (no live services needed)
	@echo "=== 1/3  Unit tests ==="
	$(PYTEST) tests/unit/ -q --no-header
	@echo "=== 2/3  Regulatory demo (mock mode) ==="
	$(PYTHON) scripts/demo_regulatory.py
	@echo "=== 3/3  API import check ==="
	$(PYTHON) -c "from api.main import app; print('API: ok')"
	@echo "=== Smoke test passed ==="

# kg_backup.py takes a required subcommand (backup | restore | list); both of
# these targets previously omitted it and died at argparse, and backup-s3
# passed a --s3-bucket flag that does not exist — S3 is addressed through the
# output path.
backup:        ## Export full graph to NDJSON (TENANT= required, OUTPUT= optional)
	$(PYTHON) scripts/kg_backup.py backup --tenant $(TENANT) --output $(or $(OUTPUT),./backup)

backup-s3:     ## Export graph to S3 (TENANT= S3_BUCKET= required, S3_PREFIX= optional)
	$(PYTHON) scripts/kg_backup.py backup --tenant $(TENANT) \
	  --output s3://$(S3_BUCKET)/$(or $(S3_PREFIX),graphrag)

restore:       ## Restore a graph from NDJSON (TENANT= INPUT= required)
	$(PYTHON) scripts/kg_backup.py restore --tenant $(TENANT) --input $(INPUT)

test-e2e:      ## Run live end-to-end tests (requires Docker for testcontainers)
	$(PYTEST) tests/e2e/ -v
