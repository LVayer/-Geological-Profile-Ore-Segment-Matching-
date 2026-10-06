@echo off
cd /d "%~dp0"
set "TEMP=%~dp0..\输出数据\临时"
set "TMP=%~dp0..\输出数据\临时"
if not exist "%TEMP%" mkdir "%TEMP%"
set "PYTHONDONTWRITEBYTECODE=1"
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
  echo Python environment missing. Run setup.ps1 first.
  pause
  exit /b 1
)
start "" "%~dp0.venv\Scripts\pythonw.exe" -B "%~dp0launch_gui.pyw"
