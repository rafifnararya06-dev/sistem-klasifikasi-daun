@echo off
title Hentikan Server Klasifikasi Daun
cd /d "%~dp0"
echo ============================================================
echo        MENGHENTIKAN SERVER KLASIFIKASI CITRA DAUN
echo ============================================================
echo.
echo Mencari proses di port 5000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000') do (
    echo Menghentikan PID %%a...
    taskkill /F /PID %%a 2>nul
)
echo.
echo Server telah berhasil dihentikan!
timeout /t 3
