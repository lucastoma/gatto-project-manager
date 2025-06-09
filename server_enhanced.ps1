# Enhanced PowerShell Server Manager for GattoNero AI Assistant
# Version: 2.0.0
# Provides advanced server management with auto-restart, monitoring, and production features

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("start", "stop", "status", "restart", "test", "logs", "health", "monitor", "metrics", "dashboard", "auto")]
    [string]$Command = "help",
    
    [Parameter(Mandatory=$false)]
    [switch]$Background,
    
    [Parameter(Mandatory=$false)]
    [switch]$AutoRestart,
    
    [Parameter(Mandatory=$false)]
    [int]$Port = 5000,
    
    [Parameter(Mandatory=$false)]
    [string]$Environment = "development"
)

# Configuration
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$SERVER_MANAGER = Join-Path $SCRIPT_DIR "server_manager_enhanced.py"
$LOG_FILE = Join-Path $SCRIPT_DIR "logs\server_manager.log"
$PID_FILE = Join-Path $SCRIPT_DIR "server.pid"

# Colors for output
$GREEN = "Green"
$RED = "Red"
$YELLOW = "Yellow"
$CYAN = "Cyan"
$MAGENTA = "Magenta"

function Write-Status {
    param($Message, $Color = "White")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor $Color
}

function Show-Help {
    Write-Host "🚀 Enhanced GattoNero AI Assistant Server Manager v2.0.0" -ForegroundColor $CYAN
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor $YELLOW
    Write-Host "  .\server_enhanced.ps1 [COMMAND] [OPTIONS]"
    Write-Host ""
    Write-Host "COMMANDS:" -ForegroundColor $YELLOW
    Write-Host "  start       - Start server (with optional auto-restart)"
    Write-Host "  stop        - Gracefully stop server"
    Write-Host "  status      - Show detailed server status"
    Write-Host "  restart     - Restart server"
    Write-Host "  health      - Check server health with metrics"
    Write-Host "  logs        - Show recent server logs"
    Write-Host "  monitor     - Real-time monitoring dashboard"
    Write-Host "  metrics     - Show performance metrics"
    Write-Host "  dashboard   - Open web management dashboard"
    Write-Host "  auto        - Start with auto-restart and monitoring"
    Write-Host "  test        - Run API tests"
    Write-Host ""
    Write-Host "OPTIONS:" -ForegroundColor $YELLOW
    Write-Host "  -Background   - Run in background"
    Write-Host "  -AutoRestart  - Enable automatic restart on failure"
    Write-Host "  -Port         - Server port (default: 5000)"
    Write-Host "  -Environment  - Environment (development/production)"
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor $GREEN
    Write-Host "  .\server_enhanced.ps1 start -AutoRestart -Background"
    Write-Host "  .\server_enhanced.ps1 monitor"
    Write-Host "  .\server_enhanced.ps1 auto"
    Write-Host "  .\server_enhanced.ps1 dashboard"
}

function Test-ServerRunning {
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 5 -UseBasicParsing
        return $true
    } catch {
        return $false
    }
}

function Get-ServerPID {
    if (Test-Path $PID_FILE) {
        try {
            $pid = Get-Content $PID_FILE -ErrorAction SilentlyContinue
            if ($pid -and (Get-Process -Id $pid -ErrorAction SilentlyContinue)) {
                return $pid
            }
        } catch {}
    }
    return $null
}

function Start-Server {
    if (Test-ServerRunning) {
        Write-Status "✅ Server is already running on port $Port" $GREEN
        return
    }
    
    Write-Status "🚀 Starting Enhanced Flask Server..." $CYAN
    
    if ($AutoRestart) {
        Write-Status "🔄 Auto-restart enabled" $YELLOW
        & python $SERVER_MANAGER start --auto-restart --port $Port --environment $Environment
    } else {
        & python $SERVER_MANAGER start --port $Port --environment $Environment
    }
    
    # Wait and verify
    Start-Sleep -Seconds 3
    if (Test-ServerRunning) {
        Write-Status "✅ Server started successfully on http://127.0.0.1:$Port" $GREEN
        if ($AutoRestart) {
            Write-Status "🛡️  Auto-restart monitoring active" $YELLOW
        }
    } else {
        Write-Status "❌ Failed to start server" $RED
    }
}

function Stop-Server {
    Write-Status "🛑 Stopping server..." $YELLOW
    & python $SERVER_MANAGER stop
    
    # Verify shutdown
    Start-Sleep -Seconds 2
    if (-not (Test-ServerRunning)) {
        Write-Status "✅ Server stopped successfully" $GREEN
    } else {
        Write-Status "⚠️  Server may still be running" $YELLOW
    }
}

function Show-Status {
    Write-Status "📊 Server Status Report" $CYAN
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor $CYAN
    
    & python $SERVER_MANAGER status
    
    $pid = Get-ServerPID
    if ($pid) {
        Write-Status "🔧 Process ID: $pid" $CYAN
        try {
            $process = Get-Process -Id $pid
            $uptime = (Get-Date) - $process.StartTime
            Write-Status "⏱️  Uptime: $($uptime.ToString('dd\.hh\:mm\:ss'))" $CYAN
            Write-Status "💾 Memory: $([math]::Round($process.WorkingSet64/1MB, 2)) MB" $CYAN
        } catch {
            Write-Status "⚠️  Cannot retrieve process details" $YELLOW
        }
    }
}

function Show-Health {
    Write-Status "🏥 Health Check" $CYAN
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor $CYAN
    & python $SERVER_MANAGER health
}

function Show-Logs {
    Write-Status "📋 Recent Server Logs" $CYAN
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor $CYAN
    & python $SERVER_MANAGER logs
}

function Start-Monitor {
    Write-Status "📈 Starting Real-time Monitor..." $CYAN
    Write-Status "Press Ctrl+C to stop monitoring" $YELLOW
    Write-Host ""
    
    try {
        while ($true) {
            Clear-Host
            Write-Host "🖥️  GattoNero AI Assistant - Real-time Monitor" -ForegroundColor $MAGENTA
            Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor $MAGENTA
            Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor $CYAN
            Write-Host ""
            
            Show-Status
            Write-Host ""
            Show-Health
            
            Write-Host ""
            Write-Host "⏱️  Next update in 10 seconds... (Ctrl+C to stop)" -ForegroundColor $YELLOW
            Start-Sleep -Seconds 10
        }
    } catch {
        Write-Status "📈 Monitoring stopped" $YELLOW
    }
}

function Show-Metrics {
    Write-Status "📊 Performance Metrics" $CYAN
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor $CYAN
    & python $SERVER_MANAGER metrics
}

function Open-Dashboard {
    $dashboardUrl = "http://127.0.0.1:$Port/admin/dashboard"
    if (Test-ServerRunning) {
        Write-Status "🌐 Opening management dashboard..." $CYAN
        Start-Process $dashboardUrl
    } else {
        Write-Status "❌ Server is not running. Start server first." $RED
    }
}

function Start-Auto {
    Write-Status "🤖 Starting Auto Mode..." $CYAN
    Write-Status "   - Auto-restart enabled" $YELLOW
    Write-Status "   - Background monitoring" $YELLOW
    Write-Status "   - Health checks" $YELLOW
    
    Start-Server -AutoRestart
    Start-Sleep -Seconds 5
    Start-Monitor
}

function Run-Tests {
    if (-not (Test-ServerRunning)) {
        Write-Status "❌ Server is not running. Start server first." $RED
        return
    }
    
    Write-Status "🧪 Running API Tests..." $CYAN
    & python $SERVER_MANAGER test
}

# Main execution logic
switch ($Command.ToLower()) {
    "start" { Start-Server }
    "stop" { Stop-Server }
    "status" { Show-Status }
    "restart" { 
        Stop-Server
        Start-Sleep -Seconds 3
        Start-Server
    }
    "health" { Show-Health }
    "logs" { Show-Logs }
    "monitor" { Start-Monitor }
    "metrics" { Show-Metrics }
    "dashboard" { Open-Dashboard }
    "auto" { Start-Auto }
    "test" { Run-Tests }
    default { Show-Help }
}
