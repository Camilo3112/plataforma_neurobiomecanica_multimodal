@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo CST COLOREADA V4 - AJUSTE SAGITAL Y CORONAL
echo ============================================================
echo.

where py >nul 2>&1
if not errorlevel 1 goto RUN_PY

where python >nul 2>&1
if not errorlevel 1 goto RUN_PYTHON

echo No se encontro Python en el PATH.
set "EXITCODE=1"
goto FIN

:RUN_PY
py -3 "%~dp0extraer_visualizar_cst_colores_V4_coronal.py"
set "EXITCODE=%ERRORLEVEL%"
goto FIN

:RUN_PYTHON
python "%~dp0extraer_visualizar_cst_colores_V4_coronal.py"
set "EXITCODE=%ERRORLEVEL%"

:FIN
echo.
if "%EXITCODE%"=="0" (
    echo PROCESO TERMINADO CORRECTAMENTE.
) else (
    echo EL PROCESO TERMINO CON ERROR: %EXITCODE%
)
echo.
pause
exit /b %EXITCODE%
