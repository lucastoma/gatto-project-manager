@echo off
REM Enhanced Flask Server Manager for GattoNero AI Assistant
REM Usage: server.cmd [start|stop|status|restart|test|logs|health]

if "%1"=="" (
    echo Usage: server.cmd [start^|stop^|status^|restart^|test^|logs^|health]
    echo.
    echo Commands:
    echo   start    - Start server in background
    echo   stop     - Stop server
    echo   status   - Show server status
    echo   restart  - Restart server
    echo   test     - Run tests ^(requires running server^)
    echo   logs     - Show recent logs
    echo   health   - Check server health
    exit /b 1
)

python server_manager.py %1
