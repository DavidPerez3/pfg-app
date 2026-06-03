param(
    [string]$PostgresUrl = "postgresql+psycopg://pfg:pfg_dev_password@localhost:5432/pfg_app",
    [string]$SQLitePath = "..\\app_state.db"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Split-Path -Parent $scriptDir
$python = Join-Path $backendDir ".venv\\Scripts\\python.exe"
$sqliteFullPath = (Resolve-Path (Join-Path $scriptDir $SQLitePath)).Path

Write-Host "Verifying PostgreSQL target..."
& $python (Join-Path $scriptDir "verify_postgres_store.py") --database-url $PostgresUrl

Write-Host "Migrating SQLite app state into PostgreSQL..."
& $python (Join-Path $scriptDir "migrate_sqlite_to_postgres.py") --sqlite-path $sqliteFullPath --postgres-url $PostgresUrl

Write-Host ""
Write-Host "Cutover prepared."
Write-Host "Next manual step: set APP_STATE_DATABASE_URL in backend/.env and restart the backend."
