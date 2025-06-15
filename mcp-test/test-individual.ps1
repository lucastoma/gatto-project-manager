# Test individual MCP tool

param (
    [Parameter(Mandatory=$true)]
    [string]$Method,
    
    [Parameter(Mandatory=$false)]
    [string]$ParamsJson = "{}"
)

# Test variables
$SERVER_PATH = "D:/projects/gatto-ps-ai-link1/gatto_filesystem_v2/dist/server/index.js"
$TEST_DIR = "D:/projects/gatto-ps-ai-link1/mcp-test"

# Parse params
$params = $ParamsJson | ConvertFrom-Json

# Create request
$request = @{
    jsonrpc = "2.0"
    id = 1
    method = $Method
    params = $params
} | ConvertTo-Json -Compress

$bytes = [System.Text.Encoding]::UTF8.GetBytes($request)
$frame = "Content-Length: $($bytes.Length)`r`n`r`n$request"

Write-Host "Request: $request" -ForegroundColor DarkGray
Write-Host "Testing $Method..." -ForegroundColor Cyan

# Execute request
$result = $frame | node $SERVER_PATH $TEST_DIR

Write-Host "Response:" -ForegroundColor Yellow
Write-Host $result