@echo off
setlocal
cd /d "%~dp0"
echo ============================================================================
echo   WeaveForge Studio — 60-120+ FPS 3D Multi-Scale Physics Sandbox (Rig 1)
echo ============================================================================

if not exist ".venv\Scripts\python.exe" (
    echo [Setup] Creating Python virtual environment and installing dependencies...
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -e ".[dev]"
)

".venv\Scripts\python.exe" -m weaveforge studio %*
if errorlevel 1 (
    echo.
    echo [WeaveForge] Studio exited with an error. Press any key to close...
    pause >nul
)
