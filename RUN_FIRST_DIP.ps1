$ErrorActionPreference = 'Stop'
$script = Join-Path $PSScriptRoot 'test_first_dip.py'
$python = 'D:\egx\Rashid\.venv312\Scripts\python.exe'
& $python $script --data 'D:\egx\metals_daily_4000\data' --out 'D:\egx\metals_daily_4000\RS3_FIRST_DIP_RESULTS'
if ($LASTEXITCODE -eq 0) { Start-Process 'D:\egx\metals_daily_4000\RS3_FIRST_DIP_RESULTS\FIRST_DIP_REPORT.html' }
