@echo off
if "%~1"=="" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" -Mode kvmem -ViewTokens 32768
) else (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" -Mode kvmem -ViewTokens 32768 -Model "%~1"
)
