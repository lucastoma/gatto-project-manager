# Enhanced Flask Server Manager for GattoNero AI Assistant
# Usage: .\server.ps1 [start|stop|status|restart|test|logs|health]

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("start", "stop", "status", "restart", "test", "logs", "health")]
    [string]$Command
)

if (-not $Command) {
    Write-Host "Usage: .\server.ps1 [start|stop|status|restart|test|logs|health]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  start    - Start server in background"
    Write-Host "  stop     - Stop server"
    Write-Host "  status   - Show server status"
    Write-Host "  restart  - Restart server"
    Write-Host "  test     - Run tests (requires running server)"
    Write-Host "  logs     - Show recent logs"
    Write-Host "  health   - Check server health"
    exit 1
}

# Run the Python server manager
python server_manager.py $Command
