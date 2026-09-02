[CmdletBinding()]
param(
    [ValidateRange(0.1, 60.0)]
    [double]$IntervalSeconds = 0.5
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $ProjectRoot

Write-Host 'Avvio paper engine headless: MT5 + inferenza + ledger, senza Streamlit/Chrome.'
Write-Host 'Non avviare contemporaneamente la pagina Live Paper: userebbe lo stesso ledger.'
python -m src.paper.headless --project-root $ProjectRoot --interval-seconds $IntervalSeconds
