# start_demo.ps1 — kills stale API/worker processes and starts fresh ones
# Usage: .\start_demo.ps1

$root = $PSScriptRoot
$python = "$root\.venv\Scripts\python.exe"

# ── Kill any stale API / worker processes ──────────────────────────────────────
Write-Host "Killing stale processes..."
Get-WmiObject Win32_Process | Where-Object {
    ($_.CommandLine -like "*uvicorn*" -or $_.CommandLine -like "*query_worker*") -and
    $_.CommandLine -notlike "*powershell*"
} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2

# ── Start API ──────────────────────────────────────────────────────────────────
Write-Host "Starting API on :8000..."
$apiEnv = @{
    GRAPHRAG_DEFAULT_TENANT = "automotive"
    REDIS_URL               = "redis://127.0.0.1:6379/0"
    ENV                     = "development"
    PYTHONUTF8              = "1"
}
$apiProcess = Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "api.main:app", "--port", "8000" `
    -WorkingDirectory $root `
    -Environment $apiEnv `
    -RedirectStandardOutput "$root\logs\api.log" `
    -RedirectStandardError  "$root\logs\api.log" `
    -PassThru -WindowStyle Hidden
Write-Host "  API PID: $($apiProcess.Id)"

# ── Start worker ───────────────────────────────────────────────────────────────
Write-Host "Starting query worker..."
$workerProcess = Start-Process -FilePath $python `
    -ArgumentList "workers\query_worker.py" `
    -WorkingDirectory $root `
    -Environment $apiEnv `
    -RedirectStandardOutput "$root\logs\worker.log" `
    -RedirectStandardError  "$root\logs\worker.log" `
    -PassThru -WindowStyle Hidden
Write-Host "  Worker PID: $($workerProcess.Id)"

# ── Wait for API to be ready ───────────────────────────────────────────────────
Write-Host "Waiting for API..."
$ready = $false
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8000/health" -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
}

if ($ready) {
    Write-Host "Ready! Open: http://localhost:8000/demo" -ForegroundColor Green
} else {
    Write-Host "API did not start — check logs\api.log" -ForegroundColor Red
}
