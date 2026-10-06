@echo off
if "%~1"=="" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" -Mode dense
) else (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" -Mode dense -Model "%~1"
)
