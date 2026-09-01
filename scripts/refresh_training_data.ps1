[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $ProjectRoot

& powershell -ExecutionPolicy Bypass -File .\scripts\update_history.ps1
if ($LASTEXITCODE -ne 0) { throw "Aggiornamento storico fallito: $LASTEXITCODE" }

python -c "from pathlib import Path; from src.modeling.training_snapshot import create_training_snapshot; print(create_training_snapshot(Path.cwd()))"
if ($LASTEXITCODE -ne 0) { throw "Creazione snapshot training fallita: $LASTEXITCODE" }
