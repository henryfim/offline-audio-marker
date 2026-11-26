@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Setting up the offline analysis environment...
  powershell -ExecutionPolicy Bypass -File "%~dp0install_dependencies.ps1"
  if errorlevel 1 pause & exit /b 1
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
