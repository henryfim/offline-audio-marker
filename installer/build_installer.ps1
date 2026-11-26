param(
    [string] $Wix = 'wix',
    [string] $Version = '1.0.0',
    [string] $OutputDirectory = (Join-Path $PSScriptRoot '..\release')
)

$ErrorActionPreference = 'Stop'
$appSource = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\dist-public\OfflineAudioMarker')).ProviderPath
$sourceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
foreach ($asset in @('THIRD_PARTY_NOTICES.md', 'third-party-licenses')) {
    Copy-Item -LiteralPath (Join-Path $sourceRoot $asset) -Destination $appSource -Recurse -Force
}
$required = @('OfflineAudioMarker.exe', 'THIRD_PARTY_NOTICES.md')
foreach ($file in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $appSource $file) -PathType Leaf)) {
        throw "Missing package file $file. Build the public app first."
    }
}
$unexpected = @(Get-ChildItem -LiteralPath $appSource -Recurse -File -Force | Where-Object {
    $_.Name -in @('UVR_MDXNET_1_9703.onnx', 'reference_annotations.json') -or $_.Name -like 'ffmpeg*.exe'
})
if ($unexpected.Count -or (Test-Path -LiteralPath (Join-Path $appSource 'Analysis Results'))) {
    throw 'Build the public distribution before creating an installer.'
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$msi = Join-Path (Resolve-Path -LiteralPath $OutputDirectory).ProviderPath 'OfflineAudioMarker-Setup-x64.msi'
& $Wix build (Join-Path $PSScriptRoot 'OfflineAudioMarker.wxs') -arch x64 -pdbtype none -d "AppSource=$appSource" -d "AppVersion=$Version" -out $msi
if ($LASTEXITCODE -ne 0) { throw "WiX failed with exit code $LASTEXITCODE." }
$digest = (Get-FileHash -LiteralPath $msi -Algorithm SHA256).Hash.ToLowerInvariant()
"$digest  $(Split-Path -Leaf $msi)" | Set-Content -LiteralPath "$msi.sha256" -Encoding ascii
Get-Item -LiteralPath $msi, "$msi.sha256"

