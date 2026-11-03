@echo off
setlocal
cd /d "%~dp0"
echo ============================================================================
echo   WeaveForge — Automated Symbolic Proofs, Red-Team Audit ^& Pytest Suite
echo ============================================================================

if not exist ".venv\Scripts\python.exe" (
    echo [Setup] Creating Python virtual environment and installing dependencies...
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -e ".[dev]"
)

".venv\Scripts\python.exe" -m weaveforge audit
if errorlevel 1 (
    echo.
    echo [FAIL] WeaveForge symbolic audit failed!
    if /i not "%~1"=="--no-pause" pause
    exit /b 1
)
echo.
".venv\Scripts\python.exe" -m pytest -v
if errorlevel 1 (
    echo.
    echo [FAIL] One or more verification tests failed!
    if /i not "%~1"=="--no-pause" pause
    exit /b 1
)
echo.
echo [PASS] All WeaveForge symbolic proofs, GPU parity checks, and crucible tests passed!
if /i not "%~1"=="--no-pause" pause
