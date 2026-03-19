@echo off
title RND - Rapid Neural Designer
echo Starting Rapid Neural Designer...
echo.

cd /d "%~dp0web_interface"

echo Starting web server on port 8089...
start "RND Web Server" /min cmd /c "python -m http.server 8089 --bind 0.0.0.0"

echo Starting backend on port 5000...
start "RND Backend" /min cmd /c "python backend.py"

timeout /t 2 /nobreak >nul
echo.
echo RND is running:
echo   Web:     http://localhost:8089
echo   Backend: http://localhost:5000
echo.
echo Press any key to stop all servers...
pause >nul

echo Stopping servers...
taskkill /fi "WINDOWTITLE eq RND Web Server" /f >nul 2>&1
taskkill /fi "WINDOWTITLE eq RND Backend" /f >nul 2>&1
echo Done.
