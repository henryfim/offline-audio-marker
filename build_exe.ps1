param(
    [switch] $IncludeLocalAssets
)

$ErrorActionPreference = 'Stop'
$appRoot = $PSScriptRoot
$venv = Join-Path $appRoot '.venv-build'
$python = Join-Path $venv 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    & py -3.13 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create an isolated Python 3.13 build environment.' }
}
$config = Get-Content -LiteralPath (Join-Path $venv 'pyvenv.cfg')
if ($config -notcontains 'include-system-site-packages = false') {
    throw "The build environment at $venv must be isolated from global site-packages."
}

$models = @('yamnet.tflite', 'local_style.json')
$distDir = Join-Path $appRoot 'dist-public'
if ($IncludeLocalAssets) {
    $models += @('UVR_MDXNET_1_9703.onnx', 'reference_annotations.json')
    $distDir = Join-Path $appRoot 'dist-local'
}
foreach ($model in $models) {
    if (-not (Test-Path -LiteralPath (Join-Path $appRoot "models\$model") -PathType Leaf)) {
        throw "Missing models\$model."
    }
}

& $python -I -m pip install -r (Join-Path $appRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed (exit $LASTEXITCODE)." }

$previous = $env:OFFLINE_AUDIO_MARKER_LOCAL_ASSETS
$env:OFFLINE_AUDIO_MARKER_LOCAL_ASSETS = if ($IncludeLocalAssets) { '1' } else { '0' }
try {
    & $python -I -m PyInstaller --clean --noconfirm --distpath $distDir --workpath (Join-Path $appRoot 'build') (Join-Path $appRoot 'OfflineAudioMarker.spec')
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit $LASTEXITCODE)." }
    Write-Host "Build completed. EXE is in $distDir\OfflineAudioMarker\OfflineAudioMarker.exe" -ForegroundColor Green
}
finally {
    $env:OFFLINE_AUDIO_MARKER_LOCAL_ASSETS = $previous
}
