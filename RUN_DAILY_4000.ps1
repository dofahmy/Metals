$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = 'D:\egx\Rashid\.venv312\Scripts\python.exe'
if (!(Test-Path $python)) { $python = 'py' }
& $python -u "$PSScriptRoot\metals_daily_4000.py" --out "D:\egx\metals_daily_4000" --bars 4000 --rs-bars 20 --near-days 3
if ($LASTEXITCODE -ne 0) { Write-Host 'Download or analysis failed. See errors above.'; exit 1 }
Start-Process 'D:\egx\metals_daily_4000\DAILY_RS_COMPARISON.html'
