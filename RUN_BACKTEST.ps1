$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = 'D:\egx\Rashid\.venv312\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'py' }
if ($python -eq 'py') { & py -3 .\rs_backtest_abc.py @args } else { & $python .\rs_backtest_abc.py @args }
if ($LASTEXITCODE -ne 0) { throw "Backtest failed. Exit code $LASTEXITCODE" }
Start-Process 'D:\egx\metals_all4_daily\RS_ABC_BACKTEST\BACKTEST_REPORT.html'
