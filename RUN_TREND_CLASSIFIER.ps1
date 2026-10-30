$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$venv = 'D:\egx\Rashid\.venv312\Scripts\python.exe'
if (Test-Path $venv) { $py = $venv } else { $py = 'py' }
& $py -u "$PSScriptRoot\ma_touch_trend_classifier.py" --open
if ($LASTEXITCODE -ne 0) { Write-Host 'FAILED. Verify daily OHLC CSV paths.'; exit $LASTEXITCODE }
