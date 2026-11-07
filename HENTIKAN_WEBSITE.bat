@echo off
title Hentikan Server Klasifikasi Daun
cd /d "%~dp0"
cls
echo ============================================================
echo        MENGHENTIKAN SERVER KLASIFIKASI CITRA DAUN
echo ============================================================
echo.
echo 1. Menghentikan proses Python di port 5000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000') do (
    taskkill /F /PID %%a 2>nul
)
echo 2. Menghentikan seluruh proses python.exe dan pythonw.exe...
taskkill /F /IM python.exe 2>nul
taskkill /F /IM pythonw.exe 2>nul
echo.
echo Server telah berhasil dihentikan sepenuhnya!
timeout /t 2 >nul
