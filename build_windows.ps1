<#
  Build S3eidat Enhancer for Windows.

    powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
    powershell -ExecutionPolicy Bypass -File .\build_windows.ps1 -SkipTests

  Produces:
    dist\S3eidat Enhancer\S3eidat Enhancer.exe   (one-folder app)
    release\S3eidat-Enhancer-Windows.zip         (the same folder, zipped)
  The AI models are downloaded by the app on first run; nothing here publishes anything.
#>
param(
    [string]$FfmpegDir = "C:\ffmpeg\ffmpeg-2025-10-09-git-469aad3897-full_build\bin",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$py = ".venv\Scripts\python.exe"
$name = "S3eidat Enhancer"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

Step "Checking Python packages"
& $py -c "import webview, PyInstaller, onnxruntime, cv2, PIL"
if ($LASTEXITCODE -ne 0) { throw "run: $py -m pip install -r requirements.txt pyinstaller" }

Step "Bundling ffmpeg + ffprobe from $FfmpegDir"
New-Item -ItemType Directory -Force "bin\win" | Out-Null
foreach ($exe in @("ffmpeg.exe", "ffprobe.exe")) {
    $src = Join-Path $FfmpegDir $exe
    $dst = Join-Path "bin\win" $exe
    if (-not (Test-Path $src)) { throw "Missing $src - pass -FfmpegDir pointing at a folder with ffmpeg.exe and ffprobe.exe" }
    if (-not (Test-Path $dst) -or (Get-Item $src).Length -ne (Get-Item $dst).Length) { Copy-Item $src $dst -Force }
}

Step "Drawing the icon"
& $py -I tools\make_icon.py
if ($LASTEXITCODE -ne 0) { throw "icon failed" }

if (-not $SkipTests) {
    Step "Running tests"
    & $py -m pytest -q -p no:cacheprovider
    if ($LASTEXITCODE -ne 0) { throw "tests failed" }
}

Step "Building with PyInstaller (one-folder)"
Get-Process $name -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
& $py -m PyInstaller --noconfirm --clean --log-level WARN Enhancer.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

Step "Zipping"
New-Item -ItemType Directory -Force "release" | Out-Null
$zip = "release\S3eidat-Enhancer-Windows.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path "dist\$name" -DestinationPath $zip -CompressionLevel Optimal
$mb = [math]::Round((Get-Item $zip).Length / 1MB)
Write-Host "`nDone: dist\$name\$name.exe  ($zip, $mb MB)" -ForegroundColor Green
