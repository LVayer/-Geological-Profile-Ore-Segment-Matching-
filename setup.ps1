$ErrorActionPreference = 'Stop'
$env:TEMP = Join-Path $PSScriptRoot '..\输出数据\临时'
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
if (-not (Test-Path "$PSScriptRoot\.venv\Scripts\python.exe")) {
    python -B -m venv "$PSScriptRoot\.venv"
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
& "$PSScriptRoot\.venv\Scripts\python.exe" -B -m pip install --no-cache-dir -r "$PSScriptRoot\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw 'dependency installation failed' }
