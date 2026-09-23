# AI Knowledge Graph Platform - Fly.io Deployment Script
# Run from: fly\
# Usage: .\deploy.ps1
#
# Required environment variables (set before running — none of these are
# committed anywhere, and the data stores no longer ship with a default
# password):
#   GOOGLE_API_KEY, JWT_SECRET_KEY, GOOGLE_OAUTH_CLIENT_ID,
#   GOOGLE_OAUTH_CLIENT_SECRET, NEO4J_PASSWORD, RABBITMQ_PASSWORD,
#   REDIS_PASSWORD

$ROOT = Split-Path -Parent $PSScriptRoot

foreach ($var in @(
    "GOOGLE_API_KEY", "JWT_SECRET_KEY", "GOOGLE_OAUTH_CLIENT_ID",
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
Set-Location "$PSScriptRoot\api"
flyctl apps create graphrag-api --machines 2>$null
flyctl volumes create kpi_data --size 1 --region ams -a graphrag-api 2>$null
flyctl secrets set `
  GOOGLE_API_KEY="$env:GOOGLE_API_KEY" `
  JWT_SECRET_KEY="$env:JWT_SECRET_KEY" `
  GOOGLE_OAUTH_CLIENT_ID="$env:GOOGLE_OAUTH_CLIENT_ID" `
  GOOGLE_OAUTH_CLIENT_SECRET="$env:GOOGLE_OAUTH_CLIENT_SECRET" `
  NEO4J_URI="bolt://graphrag-neo4j.internal:7687" `
  NEO4J_USER="neo4j" `
  NEO4J_PASSWORD="$env:NEO4J_PASSWORD" `
  RABBITMQ_URL="amqp://graphrag:$env:RABBITMQ_PASSWORD@graphrag-rabbit.internal:5672/" `
  REDIS_URL="redis://:$env:REDIS_PASSWORD@graphrag-redis-sn.internal:6379/0" `
  KPI_DB_PATH="/data/kpis.db" `
  CORS_ORIGINS='["https://graphrag-api.fly.dev","https://graphrag-dashboard.fly.dev"]' `
  -a graphrag-api
flyctl deploy --ha=false -c "$PSScriptRoot\api\fly.toml" --dockerfile "$ROOT\Dockerfile" --path "$ROOT"

Write-Host "`n=== Step 5: Deploy Workers (ingestion + query) ===" -ForegroundColor Cyan
Set-Location "$PSScriptRoot\workers"
flyctl apps create graphrag-workers --machines 2>$null
flyctl volumes create kpi_data --size 1 --region ams -a graphrag-workers 2>$null
flyctl secrets set `
  GOOGLE_API_KEY="$env:GOOGLE_API_KEY" `
  NEO4J_URI="bolt://graphrag-neo4j.internal:7687" `
  NEO4J_USER="neo4j" `
  NEO4J_PASSWORD="$env:NEO4J_PASSWORD" `
  RABBITMQ_URL="amqp://graphrag:$env:RABBITMQ_PASSWORD@graphrag-rabbit.internal:5672/" `
  REDIS_URL="redis://:$env:REDIS_PASSWORD@graphrag-redis-sn.internal:6379/0" `
  KPI_DB_PATH="/data/kpis.db" `
  -a graphrag-workers
flyctl deploy --ha=false -c "$PSScriptRoot\workers\fly.toml" --dockerfile "$ROOT\Dockerfile" --path "$ROOT"

Write-Host "`n=== Step 6: Deploy Evaluation Worker ===" -ForegroundColor Cyan
Set-Location "$PSScriptRoot\evaluation"
flyctl apps create graphrag-evaluation --machines 2>$null
flyctl volumes create kpi_data --size 1 --region ams -a graphrag-evaluation 2>$null
flyctl secrets set `
  GOOGLE_API_KEY="$env:GOOGLE_API_KEY" `
  NEO4J_URI="bolt://graphrag-neo4j.internal:7687" `
  NEO4J_USER="neo4j" `
  NEO4J_PASSWORD="$env:NEO4J_PASSWORD" `
  RABBITMQ_URL="amqp://graphrag:$env:RABBITMQ_PASSWORD@graphrag-rabbit.internal:5672/" `
  REDIS_URL="redis://:$env:REDIS_PASSWORD@graphrag-redis-sn.internal:6379/0" `
  KPI_DB_PATH="/data/kpis.db" `
  -a graphrag-evaluation
flyctl deploy --ha=false -c "$PSScriptRoot\evaluation\fly.toml" --dockerfile "$ROOT\Dockerfile" --path "$ROOT"

Write-Host "`n=== Step 7: Deploy Dashboard ===" -ForegroundColor Cyan
Set-Location "$PSScriptRoot\dashboard"
flyctl apps create graphrag-dashboard --machines 2>$null
flyctl volumes create kpi_data --size 1 --region ams -a graphrag-dashboard 2>$null
flyctl secrets set `
  KPI_DB_PATH="/data/kpis.db" `
  -a graphrag-dashboard
flyctl deploy --ha=false -c "$PSScriptRoot\dashboard\fly.toml" --dockerfile "$ROOT\Dockerfile" --path "$ROOT"

Write-Host "`n=== Deployment Complete ===" -ForegroundColor Green
Write-Host "API:        https://graphrag-api.fly.dev"
Write-Host "Dashboard:  https://graphrag-dashboard.fly.dev"
Write-Host "Neo4j:      bolt://graphrag-neo4j.internal:7687 (internal only)"
Write-Host "RabbitMQ:   amqp://graphrag-rabbit.internal:5672 (internal only)"
Write-Host "Redis:      redis://graphrag-redis-sn.internal:6379 (internal only)"
