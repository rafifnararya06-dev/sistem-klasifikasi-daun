@echo off
title Sistem Klasifikasi Citra Daun - UTS Computer Vision
cd /d "%~dp0"
cls
echo ============================================================
echo      MENJALANKAN SISTEM KLASIFIKASI CITRA DAUN (UTS CV)
echo ============================================================
echo.
echo 1. Membersihkan proses lama di port 5000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000') do (
    taskkill /F /PID %%a 2>nul
)
taskkill /F /IM python.exe 2>nul
taskkill /F /IM pythonw.exe 2>nul

echo 2. Membuka peramban di http://localhost:5000...
start "" "http://localhost:5000/?v=%RANDOM%"

echo 3. Menjalankan server backend Python Flask...
echo.
echo * Jendela CMD ini harus tetap terbuka selama website digunakan.
echo * Tekan Ctrl + C atau tutup jendela ini untuk menghentikan website.
echo ============================================================
echo.
python app.py
pause
