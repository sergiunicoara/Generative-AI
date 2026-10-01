# AI Knowledge Graph Platform - Fly.io Deployment Script
# Run from: fly\
# Usage: .\deploy.ps1
#
# Required environment variables (set before running — none of these are
# committed anywhere, and the data stores no longer ship with a default
# password):
#   GROQ_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY (embeddings),
#   JWT_SECRET_KEY, SESSION_SECRET_KEY (must differ from JWT_SECRET_KEY, both
#   >= 32 bytes), GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET,
#   NEO4J_PASSWORD, RABBITMQ_PASSWORD, REDIS_PASSWORD
#
# Every service runs with ENV=production, and graphrag.core.config.Settings
# validates JWT_SECRET_KEY, SESSION_SECRET_KEY, CORS_ORIGINS, NEO4J_PASSWORD
# and RABBITMQ_URL at import time in ALL of them -- not just the API -- so each
# service below gets that full set (audit-2026-10-01.md, H7).
#
# App names must match `app = ...` in each fly/<service>/fly.toml: `flyctl
# deploy -c` deploys to the toml's app, so secrets and volumes created on any
# other name never reach the app that runs.

$ROOT = Split-Path -Parent $PSScriptRoot

# Fly app names -- keep in sync with fly/<service>/fly.toml.
$API_APP        = "graphrag-api-sn"
$WORKERS_APP    = "graphrag-workers-sn"
$EVALUATION_APP = "graphrag-evaluation-sn"
$DASHBOARD_APP  = "graphrag-dashboard-sn"
# Single-quoted JSON pieces, as the original literal was: pydantic-settings
# parses list[str] settings as JSON, and the embedded quotes must survive.
$CORS = '["https://' + $API_APP + '.fly.dev","https://' + $DASHBOARD_APP + '.fly.dev"]'

foreach ($var in @(
    "GROQ_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
    "JWT_SECRET_KEY", "SESSION_SECRET_KEY", "GOOGLE_OAUTH_CLIENT_ID",
    "GOOGLE_OAUTH_CLIENT_SECRET", "NEO4J_PASSWORD", "RABBITMQ_PASSWORD",
    "REDIS_PASSWORD")) {
  if (-not (Get-Item "env:$var" -ErrorAction SilentlyContinue)) {
    Write-Host "Missing required environment variable: $var" -ForegroundColor Red
    exit 1
  }
}

Write-Host "`n=== Step 1: Deploy Neo4j ===" -ForegroundColor Cyan
Set-Location "$PSScriptRoot\neo4j"
flyctl apps create graphrag-neo4j --machines 2>$null
flyctl volumes create neo4j_data --size 3 --region ams -a graphrag-neo4j 2>$null
flyctl secrets set NEO4J_AUTH="neo4j/$env:NEO4J_PASSWORD" -a graphrag-neo4j
flyctl deploy --ha=false

Write-Host "`n=== Step 2: Deploy RabbitMQ ===" -ForegroundColor Cyan
Set-Location "$PSScriptRoot\rabbitmq"
flyctl apps create graphrag-rabbit --machines 2>$null
flyctl volumes create rabbitmq_data --size 1 --region ams -a graphrag-rabbit 2>$null
flyctl secrets set RABBITMQ_DEFAULT_PASS="$env:RABBITMQ_PASSWORD" -a graphrag-rabbit
flyctl deploy --ha=false

Write-Host "`n=== Step 3: Deploy Redis ===" -ForegroundColor Cyan
Set-Location "$PSScriptRoot\redis"
flyctl apps create graphrag-redis-sn --machines 2>$null
flyctl secrets set REDIS_PASSWORD="$env:REDIS_PASSWORD" -a graphrag-redis-sn
flyctl deploy --ha=false

Write-Host "`n=== Step 4: Set secrets for API ===" -ForegroundColor Cyan
Set-Location "$PSScriptRootpi"
flyctl apps create $API_APP --machines 2>$null
flyctl volumes create kpi_data --size 1 --region ams -a $API_APP 2>$null
flyctl secrets set `
  GROQ_API_KEY="$env:GROQ_API_KEY" `
  DEEPSEEK_API_KEY="$env:DEEPSEEK_API_KEY" `
  OPENAI_API_KEY="$env:OPENAI_API_KEY" `
  JWT_SECRET_KEY="$env:JWT_SECRET_KEY" `
  SESSION_SECRET_KEY="$env:SESSION_SECRET_KEY" `
  GOOGLE_OAUTH_CLIENT_ID="$env:GOOGLE_OAUTH_CLIENT_ID" `
  GOOGLE_OAUTH_CLIENT_SECRET="$env:GOOGLE_OAUTH_CLIENT_SECRET" `
  NEO4J_URI="bolt://graphrag-neo4j.internal:7687" `
  NEO4J_USER="neo4j" `
  NEO4J_PASSWORD="$env:NEO4J_PASSWORD" `
  RABBITMQ_URL="amqp://graphrag:$env:RABBITMQ_PASSWORD@graphrag-rabbit.internal:5672/" `
  REDIS_URL="redis://:$env:REDIS_PASSWORD@graphrag-redis-sn.internal:6379/0" `
  KPI_DB_PATH="/data/kpis.db" `
  CORS_ORIGINS=$CORS `
  -a $API_APP
flyctl deploy --ha=false -c "$PSScriptRootpily.toml" --dockerfile "$ROOT\Dockerfile" --path "$ROOT"

Write-Host "`n=== Step 5: Deploy Workers (ingestion + query) ===" -ForegroundColor Cyan
Set-Location "$PSScriptRoot\workers"
flyctl apps create $WORKERS_APP --machines 2>$null
flyctl volumes create kpi_data --size 1 --region ams -a $WORKERS_APP 2>$null
flyctl secrets set `
  GROQ_API_KEY="$env:GROQ_API_KEY" `
  DEEPSEEK_API_KEY="$env:DEEPSEEK_API_KEY" `
  OPENAI_API_KEY="$env:OPENAI_API_KEY" `
  JWT_SECRET_KEY="$env:JWT_SECRET_KEY" `
  SESSION_SECRET_KEY="$env:SESSION_SECRET_KEY" `
  NEO4J_URI="bolt://graphrag-neo4j.internal:7687" `
  NEO4J_USER="neo4j" `
  NEO4J_PASSWORD="$env:NEO4J_PASSWORD" `
  RABBITMQ_URL="amqp://graphrag:$env:RABBITMQ_PASSWORD@graphrag-rabbit.internal:5672/" `
  REDIS_URL="redis://:$env:REDIS_PASSWORD@graphrag-redis-sn.internal:6379/0" `
  KPI_DB_PATH="/data/kpis.db" `
  CORS_ORIGINS=$CORS `
  -a $WORKERS_APP
flyctl deploy --ha=false -c "$PSScriptRoot\workersly.toml" --dockerfile "$ROOT\Dockerfile" --path "$ROOT"

Write-Host "`n=== Step 6: Deploy Evaluation Worker ===" -ForegroundColor Cyan
Set-Location "$PSScriptRoot\evaluation"
flyctl apps create $EVALUATION_APP --machines 2>$null
flyctl volumes create kpi_data --size 1 --region ams -a $EVALUATION_APP 2>$null
flyctl secrets set `
  GROQ_API_KEY="$env:GROQ_API_KEY" `
  DEEPSEEK_API_KEY="$env:DEEPSEEK_API_KEY" `
  OPENAI_API_KEY="$env:OPENAI_API_KEY" `
  JWT_SECRET_KEY="$env:JWT_SECRET_KEY" `
  SESSION_SECRET_KEY="$env:SESSION_SECRET_KEY" `
  NEO4J_URI="bolt://graphrag-neo4j.internal:7687" `
  NEO4J_USER="neo4j" `
  NEO4J_PASSWORD="$env:NEO4J_PASSWORD" `
  RABBITMQ_URL="amqp://graphrag:$env:RABBITMQ_PASSWORD@graphrag-rabbit.internal:5672/" `
  REDIS_URL="redis://:$env:REDIS_PASSWORD@graphrag-redis-sn.internal:6379/0" `
  KPI_DB_PATH="/data/kpis.db" `
  CORS_ORIGINS=$CORS `
  -a $EVALUATION_APP
flyctl deploy --ha=false -c "$PSScriptRoot\evaluationly.toml" --dockerfile "$ROOT\Dockerfile" --path "$ROOT"

Write-Host "`n=== Step 7: Deploy Dashboard ===" -ForegroundColor Cyan
Set-Location "$PSScriptRoot\dashboard"
flyctl apps create $DASHBOARD_APP --machines 2>$null
flyctl volumes create kpi_data --size 1 --region ams -a $DASHBOARD_APP 2>$null
# The dashboard imports api.auth.dependencies, so it needs the same signing
# secrets as the API to verify tokens, plus the settings Settings validates.
flyctl secrets set `
  JWT_SECRET_KEY="$env:JWT_SECRET_KEY" `
  SESSION_SECRET_KEY="$env:SESSION_SECRET_KEY" `
  NEO4J_PASSWORD="$env:NEO4J_PASSWORD" `
  RABBITMQ_URL="amqp://graphrag:$env:RABBITMQ_PASSWORD@graphrag-rabbit.internal:5672/" `
  REDIS_URL="redis://:$env:REDIS_PASSWORD@graphrag-redis-sn.internal:6379/0" `
  KPI_DB_PATH="/data/kpis.db" `
  CORS_ORIGINS=$CORS `
  -a $DASHBOARD_APP
flyctl deploy --ha=false -c "$PSScriptRoot\dashboardly.toml" --dockerfile "$ROOT\Dockerfile" --path "$ROOT"

Write-Host "`n=== Deployment Complete ===" -ForegroundColor Green
Write-Host "API:        https://$API_APP.fly.dev"
Write-Host "Dashboard:  https://$DASHBOARD_APP.fly.dev"
Write-Host "Neo4j:      bolt://graphrag-neo4j.internal:7687 (internal only)"
Write-Host "RabbitMQ:   amqp://graphrag-rabbit.internal:5672 (internal only)"
Write-Host "Redis:      redis://graphrag-redis-sn.internal:6379 (internal only)"
