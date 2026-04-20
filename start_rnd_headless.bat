@echo off
:: Headless startup for RND - used by Task Scheduler at boot
cd /d "C:\source\rapid-neural-designer-master\web_interface"

:: Kill any existing instances first
taskkill /f /fi "WINDOWTITLE eq RND Web Server" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq RND Backend" >nul 2>&1

:: Start web server (minimized)
start "RND Web Server" /min cmd /c "python -m http.server 8089 --bind 0.0.0.0"

:: Start backend (minimized)
start "RND Backend" /min cmd /c "python backend.py"
