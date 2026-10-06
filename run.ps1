param([string]$Command = 'demo', [string]$InputFile = '', [string]$OutputDir = '../输出数据/run')
$ErrorActionPreference = 'Stop'
$env:TEMP = Join-Path $PSScriptRoot '..\输出数据\临时'
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = '1'
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
Push-Location $PSScriptRoot
try {
    $params = @('-B','-m','stratamatch',$Command,'--output',$OutputDir)
    if ($InputFile) { $params += @('--input',$InputFile) }
    & "$PSScriptRoot\.venv\Scripts\python.exe" @params
    if ($LASTEXITCODE -ne 0) { throw "stratamatch failed: $LASTEXITCODE" }
} finally { Pop-Location }
