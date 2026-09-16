@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

net session >nul 2>&1
if not "%errorlevel%"=="0" (
    echo Solicitando permisos de administrador...
    powershell.exe -NoProfile -ExecutionPolicy Bypass ^
      -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

echo ============================================================
echo REPARAR DNS DE WSL
echo ============================================================
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass ^
  -File "%~dp0REPARAR_DNS_WSL.ps1"

set "EXITCODE=%ERRORLEVEL%"

echo.
if "%EXITCODE%"=="0" (
    echo DNS REPARADO.
    echo Ya puedes ejecutar la segmentacion V3.
) else (
    echo LA REPARACION TERMINO CON ERROR: %EXITCODE%
)

echo.
pause
exit /b %EXITCODE%
