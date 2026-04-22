@echo off
title RND Platform
echo Starting RND Platform...
echo.

cd /d "%~dp0web_interface"

echo Starting unified server on port 5000...
start "RND Platform" /min cmd /c "python backend.py"

timeout /t 3 /nobreak >nul
echo.
echo RND Platform is running:
echo   Editor:   http://localhost:5000
echo   3D View:  http://localhost:5000/3d.html
echo   API:      http://localhost:5000/api/rnd/*
echo.
echo Press any key to stop...
pause >nul

echo Stopping server...
taskkill /fi "WINDOWTITLE eq RND Platform" /f >nul 2>&1
echo Done.
