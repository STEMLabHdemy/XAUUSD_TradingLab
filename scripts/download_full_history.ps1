[CmdletBinding()]
param(
    [datetime]$StartDate = [datetime]'2003-05-05',
    [datetime]$EndDate = [datetime]::UtcNow.Date
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $ProjectRoot

New-Item -ItemType Directory -Force -Path 'data/raw/bid','data/raw/ask','data/processed','reports','logs' | Out-Null

Write-Host "XAUUSD M1 BID+ASK monthly download"
Write-Host "Range: $($StartDate.ToString('yyyy-MM-dd')) to $($EndDate.ToString('yyyy-MM-dd')) (end exclusive)"
Write-Host "Order: newest month first, then backwards toward 2003"
Write-Host "Existing valid normalized month files will be skipped; failures are retried and logged."

python -m src.data.cli --project-root $ProjectRoot download-full `
    --start $StartDate.ToString('yyyy-MM-dd') `
    --end $EndDate.ToString('yyyy-MM-dd') `
    --newest-first

if ($LASTEXITCODE -ne 0) {
    throw "Historical download stopped with exit code $LASTEXITCODE. Re-run this same command to resume."
}

Write-Host "Download completed. Build the master only after reviewing the log: logs/download_history.log"
