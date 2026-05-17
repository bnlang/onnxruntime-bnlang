# Local build (Windows). Assumes deps/windows-x64/ is already populated
# by `bnl script/install.bnl` (or hand-bootstrap).
#
# Usage:
#   .\build.ps1                build for windows-x64
#   .\build.ps1 -Clean         delete build/windows-x64
#   .\build.ps1 -Preset linux-x64 -Configure
#
# For cross-compile / other triples, prefer the preset directly:
#   cmake --preset linux-x64 && cmake --build build/linux-x64

param(
    [string]$Preset = "windows-x64",
    [switch]$Clean,
    [switch]$Configure
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$buildDir = Join-Path $PSScriptRoot "build/$Preset"
$depsDir  = Join-Path $PSScriptRoot "deps/$Preset"

if ($Clean) {
    if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }
    Write-Host "cleaned $buildDir" -ForegroundColor Yellow
    return
}

if (-not (Test-Path $depsDir)) {
    throw "deps/$Preset/ not found. Run 'bnl script/install.bnl' first (or extract the ORT prebuilt there manually)."
}

# Configure if needed (or forced)
if ($Configure -or -not (Test-Path (Join-Path $buildDir 'CMakeCache.txt'))) {
    Write-Host "configuring preset: $Preset" -ForegroundColor Cyan
    & cmake --preset $Preset
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
}

# Build
Write-Host "building: $Preset" -ForegroundColor Cyan
& cmake --build $buildDir --config Release
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

Write-Host "built: $buildDir" -ForegroundColor Green
Get-ChildItem $buildDir -Filter '*.dll' | Select-Object FullName, Length
