$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = 'D:\egx\Rashid\.venv312\Scripts\python.exe'
if (!(Test-Path $python)) { $python = 'py' }
if ($python -eq 'py') {
    & py -u .\all4_metals.py --out 'D:\egx\metals_all4_daily' --existing 'D:\egx\metals_daily_4000\data'
} else {
    & $python -u .\all4_metals.py --out 'D:\egx\metals_all4_daily' --existing 'D:\egx\metals_daily_4000\data'
}
if ($LASTEXITCODE -ne 0) { throw 'Scan failed; see error above' }
Start-Process 'D:\egx\metals_all4_daily\ALL4_COMPARISON.html'
