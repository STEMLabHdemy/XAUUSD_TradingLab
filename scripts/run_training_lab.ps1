[CmdletBinding()]
param(
    [switch]$Quick,
    [int]$Rows = 500000
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $ProjectRoot

$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$OutputRoot = Join-Path $ProjectRoot "results\training_lab\$Stamp"
$Arguments = @(
    '-m', 'src.experiments.run_cost_aware_lab',
    '--project-root', $ProjectRoot,
    '--output-root', $OutputRoot,
    '--rows', $Rows
)

if ($Quick) {
    # Fast smoke-research pass: enough to validate the pipeline, not evidence.
    $Arguments += @('--horizons', '10', '15', '--minimum-moves', '0.25', '0.50')
} else {
    # Full batch: 4 horizons × 3 executable move definitions, sequentially.
    $Arguments += @('--horizons', '5', '10', '15', '30', '--minimum-moves', '0.25', '0.50', '0.75')
}

Write-Host "Starting research-only training lab. The live paper runtime is not modified."
Write-Host "Results: $OutputRoot"
& python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Training lab stopped with exit code $LASTEXITCODE. Check $OutputRoot\lab_summary.csv"
}
Write-Host "Completed. Open: $OutputRoot\lab_summary.csv"
