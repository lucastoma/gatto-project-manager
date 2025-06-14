@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo URUCHAMIANIE SELEKTORA KONFIGURACJI...
echo.
python config-selector.py
echo.
echo Nacisnij dowolny klawisz aby zamknac...
pause >nul