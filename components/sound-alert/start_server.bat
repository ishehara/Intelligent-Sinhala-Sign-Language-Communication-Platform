@echo off
title Sound Alert API Server

echo ============================================================
echo   Sound Alert API Server — starting on port 5003
echo ============================================================

:: Activate the project venv (adjust path if yours is elsewhere)
if exist "..\..\venv\Scripts\activate.bat" (
    call ..\..\venv\Scripts\activate.bat
) else if exist "..\venv\Scripts\activate.bat" (
    call ..\venv\Scripts\activate.bat
)

:: Install Flask / flask-cors if not already present
pip install flask flask-cors --quiet

:: Run the API server from the sound-alert component root
cd /d "%~dp0"
python src\api_server.py

pause
