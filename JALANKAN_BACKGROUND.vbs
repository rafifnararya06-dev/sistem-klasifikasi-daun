Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "c:\UTS CV"

' Jalankan server di background tanpa jendela console hitam (pythonw)
WshShell.Run "pythonw app.py", 0, False

' Tunggu 1 detik agar server siap, lalu buka browser otomatis
WScript.Sleep 1200
WshShell.Run "http://localhost:5000"
