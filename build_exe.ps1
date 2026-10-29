# Build dist\TranscriptIntel\TranscriptIntel.exe and copy the data it needs next to it.
# Usage (from the project root, with the .venv active):  .\build_exe.ps1
# The .env file (API key) is deliberately NOT copied; add your own next to the .exe if you
# want Hybrid mode refreshes. Rules mode and all pre-built data work without it.
$ErrorActionPreference = "Stop"

pyinstaller TranscriptIntel.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$dist = "dist\TranscriptIntel"
New-Item -ItemType Directory -Force "$dist\data", "$dist\outputs" | Out-Null
Copy-Item -Recurse -Force "data\raw" "$dist\data\raw"
Copy-Item -Recurse -Force "data\processed" "$dist\data\processed"
# the exclusion list (and the excluded transcripts) so a copied .exe keeps them out too
if (Test-Path "data\exclusions.csv") { Copy-Item -Force "data\exclusions.csv" "$dist\data\" }
if (Test-Path "data\exclude") { Copy-Item -Recurse -Force "data\exclude" "$dist\data\exclude" }
if (Test-Path "outputs\run_summary.json") { Copy-Item -Force "outputs\run_summary.json" "$dist\outputs\" }
# the settings template (model, budget, warning share), never the real .env
Copy-Item -Force ".env.example" "$dist\.env.example"

Write-Host "Built $dist\TranscriptIntel.exe"
