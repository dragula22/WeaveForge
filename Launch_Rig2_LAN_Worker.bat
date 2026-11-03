@echo off
setlocal
cd /d "%~dp0"
echo ============================================================================
echo   WeaveForge Rig 2 LAN Compute Worker (Vulkan / DX12 wgpu — No CUDA Needed!)
echo   Optimized for AMD Radeon RX 5700 XT + Ryzen 7 3700X (or any Vulkan GPU)
echo ============================================================================

if not exist ".venv\Scripts\python.exe" (
    echo [Setup] Creating Python virtual environment and installing dependencies...
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -e ".[dev]"
)

echo.
echo [Network] Local IPv4 addresses on this machine (paste http://IP:8765 into Rig 1):
ipconfig | findstr /i "IPv4"
echo.

".venv\Scripts\python.exe" -m weaveforge worker --host 0.0.0.0 --port 8765
pause
