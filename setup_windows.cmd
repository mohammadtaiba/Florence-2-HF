@echo off
setlocal

cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Fehler: Der Python Launcher "py" wurde nicht gefunden.
    echo Installiere Python 3.11 und aktiviere "Add Python to PATH".
    exit /b 1
)

py -3.11 -m venv .venv
if errorlevel 1 exit /b 1

.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1

.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

.venv\Scripts\python.exe -m pip install -e .
if errorlevel 1 exit /b 1

echo.
echo Einrichtung abgeschlossen.
echo Lege ein Bild unter images\test.jpg ab.
echo Beispiel:
echo .venv\Scripts\python.exe -m florence2_hf.cli --image images\test.jpg --task caption
