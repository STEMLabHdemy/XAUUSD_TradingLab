[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $ProjectRoot

python -m src.data.cli --project-root $ProjectRoot update
if ($LASTEXITCODE -ne 0) {
    throw "Incremental update failed with exit code $LASTEXITCODE. See logs/download_history.log."
}

