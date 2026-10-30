$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$paths = @('D:\egx\Rashid\.venv312\Scripts\python.exe', 'D:\egx\gold_local\.venv\Scripts\python.exe')
$python = $paths | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $python) { $python = 'py' }
Write-Host "Using Python: $python"
& $python -u (Join-Path $PSScriptRoot 'build_technical_page.py') --root 'D:\egx\metals_all4_daily' --existing 'D:\egx\metals_daily_4000\data'
if ($LASTEXITCODE -ne 0) { Write-Host 'ERROR: Report was not generated.'; exit $LASTEXITCODE }
Start-Process 'D:\egx\metals_all4_daily\TECHNICAL_ALL4.html'
