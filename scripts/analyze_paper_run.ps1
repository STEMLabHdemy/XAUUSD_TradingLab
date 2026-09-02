[CmdletBinding()]
param(
    [string]$RunDirectory,
    [string]$OutputDirectory,
    [double]$MinimumMove = 0.5
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $ProjectRoot

$Arguments = @('-m', 'src.experiments.analyze_paper_run', '--project-root', $ProjectRoot)
if ($RunDirectory) { $Arguments += @('--run-dir', $RunDirectory) }
if ($OutputDirectory) { $Arguments += @('--output-root', $OutputDirectory) }
$Arguments += @('--minimum-move', $MinimumMove)

Write-Host 'Analisi read-only del Paper: il runtime live non viene modificato.'
& python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Analisi fallita con exit code $LASTEXITCODE" }
