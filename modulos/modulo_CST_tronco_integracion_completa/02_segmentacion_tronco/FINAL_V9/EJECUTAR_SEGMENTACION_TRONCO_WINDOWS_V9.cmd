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
echo SEGMENTACION REAL DEL TRONCO - WINDOWS V9
echo ============================================================
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass ^
  -File "%~dp0segmentar_tronco_todo_windows_V9.ps1"

set "EXITCODE=%ERRORLEVEL%"

echo.
if "%EXITCODE%"=="0" (
    echo PROCESO TERMINADO O ETAPA COMPLETADA.
) else (
    echo EL PROCESO TERMINO CON ERROR: %EXITCODE%
)

echo.
pause
exit /b %EXITCODE%
