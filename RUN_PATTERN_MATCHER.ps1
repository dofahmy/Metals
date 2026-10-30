$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
py .\pattern_matcher.py --open
if ($LASTEXITCODE -ne 0) { throw 'Pattern Matcher generation failed.' }
