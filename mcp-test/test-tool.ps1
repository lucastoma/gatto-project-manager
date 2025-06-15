# Poprawiony skrypt testujący dla serwera MCP filesystem

param (
    [Parameter(Mandatory=$true)]
    [string]$ToolName,
    
    [Parameter(Mandatory=$false)]
    [string]$ParamsJson = "{}"
)

# Test variables
$SERVER_PATH = "D:/projects/gatto-ps-ai-link1/gatto_filesystem_v2/dist/server/index.js"
$TEST_DIR = "D:/projects/gatto-ps-ai-link1/mcp-test"

# Parse params with error handling
$params = try {
    $ParamsJson | ConvertFrom-Json -ErrorAction Stop
} catch {
    Write-Error "Failed to parse JSON parameters: $_"
    exit 1
}

# Utwórz odpowiedni format żądania w zależności od typu narzędzia
if ($ToolName -eq "handshake" -or $ToolName -eq "list_tools") {
    # Dla handshake i list_tools używamy bezpośredniego wywołania metody
    $request = @{
        jsonrpc = "2.0"
        id = 1
        method = $ToolName
        params = $params
    } | ConvertTo-Json -Compress -Depth 10
} else {
    # Dla pozostałych narzędzi używamy call_tool
    $request = @{
        jsonrpc = "2.0"
        id = 1
        method = "call_tool"
        params = @{
            name = $ToolName
            args = $params
        }
    } | ConvertTo-Json -Compress -Depth 10
}

# Handle UTF-8 encoding properly for non-ASCII characters
$bytes = [System.Text.Encoding]::UTF8.GetBytes($request)
$frame = "Content-Length: $($bytes.Length)`r`nContent-Type: application/json; charset=utf-8`r`n`r`n$request"

Write-Host "Request: $request" -ForegroundColor DarkGray
Write-Host "Testing tool: $ToolName..." -ForegroundColor Cyan

# Execute request
$result = $frame | node $SERVER_PATH $TEST_DIR

Write-Host "Response:" -ForegroundColor Yellow
Write-Host $result