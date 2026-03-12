@echo off
REM Build DL_AnomalyDetector.exe
REM dist and build directories are fixed under dl_anomaly/

cd /d "%~dp0"
pyinstaller build.spec --noconfirm --distpath "%~dp0dist" --workpath "%~dp0build"
pause

