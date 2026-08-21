@echo off
:: 0MGE — One-click bootstrap (Windows)
set DIR=%~dp0
set VENV=%DIR%venv

echo.
echo  0MGE — Music Granular Engine
echo.

:: Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo  Python not found.
    echo  Install from: https://www.python.org/downloads/
    start https://www.python.org/downloads/
    exit /b 1
)

:: Check version
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  Python %PYVER% OK

:: Create venv
if not exist "%VENV%" (
    echo  Creating venv...
    python -m venv "%VENV%"
)
call "%VENV%\Scripts\activate.bat"

:: Install deps
echo  Checking dependencies...
pip install --quiet --upgrade pip
pip install --quiet -r "%DIR%requirements.txt"

echo.
echo  Ready!
echo.
python "%DIR%app.py"
pause
