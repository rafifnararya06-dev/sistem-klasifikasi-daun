@echo off
title Sistem Klasifikasi Citra Daun - UTS CV
cd /d "%~dp0"
cls
echo ============================================================
echo      MENJALANKAN SISTEM KLASIFIKASI CITRA DAUN
echo ============================================================
echo.
echo 1. Membuka peramban (browser) di http://localhost:5000...
start "" "http://localhost:5000"
echo 2. Menjalankan server Python Flask...
echo.
echo * Catatan: Biarkan jendela ini tetap terbuka selama memakai web.
echo * Untuk menutup aplikasi, cukup tutup jendela ini atau tekan Ctrl + C.
echo.
python app.py
pause
