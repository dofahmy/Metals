$ErrorActionPreference = 'Stop'
$py = 'D:\egx\Rashid\.venv312\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "Python not found: $py" }
& $py -u (Join-Path $PSScriptRoot 'compare_rs_days.py')
if ($LASTEXITCODE -ne 0) { throw 'Comparison failed.' }
Start-Process 'D:\egx\daily_rs_dates_comparison\RS_BREAKOUT_DATES_REPORT.html'
