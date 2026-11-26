$ErrorActionPreference = 'Stop'
$appRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $appRoot '.venv'
$python = Get-Command py -ErrorAction SilentlyContinue
if (-not $python) { throw 'Python was not found. Install Python 3.11 or newer.' }
if (-not (Test-Path (Join-Path $venv 'Scripts\python.exe'))) {
    & py -3.13 -m venv $venv
    if ($LASTEXITCODE -ne 0) { & py -3 -m venv $venv }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python virtual environment.' }
}
& (Join-Path $venv 'Scripts\python.exe') -m pip install -r (Join-Path $appRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed (exit $LASTEXITCODE)." }
foreach ($asset in @('yamnet.tflite', 'local_style.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $appRoot "models\$asset"))) {
        throw "Missing models\$asset. Restore it from the source bundle."
    }
}
Write-Host "Dependencies installed. You can run run.bat now." -ForegroundColor Green
