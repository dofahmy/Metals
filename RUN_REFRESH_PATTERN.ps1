$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
py .\refresh_price_data.py
if ($LASTEXITCODE -ne 0) {
  Write-Warning 'No symbols were refreshed. Existing CSV files were preserved.'
  exit $LASTEXITCODE
}
py .\pattern_matcher.py --open
if ($LASTEXITCODE -ne 0) { throw 'Dashboard rebuild failed.' }
