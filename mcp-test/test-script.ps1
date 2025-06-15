# Test PowerShell script to test all MCP filesystem server tools

# Test variables
$SERVER_PATH = "D:/projects/gatto-ps-ai-link1/gatto_filesystem_v2/dist/server/index.js"
$TEST_DIR = "D:/projects/gatto-ps-ai-link1/mcp-test"

# Helper function to invoke MCP
function Invoke-MCPTest {
    param (
        [string]$Method,
        [hashtable]$Params
    )
    
    $request = @{
        jsonrpc = "2.0"
        id = (Get-Random)
        method = $Method
        params = $Params
    } | ConvertTo-Json -Compress

    $bytes = [System.Text.Encoding]::UTF8.GetBytes($request)
    $frame = "Content-Length: $($bytes.Length)`r`n`r`n$request"
    
    Write-Host "`n===== Testing: $Method =====" -ForegroundColor Cyan
    Write-Host "Request: $request" -ForegroundColor DarkGray
    
    # Create a temporary file to capture output
    $tempFile = [System.IO.Path]::GetTempFileName()
    
    # Run the command and redirect output to the temp file
    $frame | node $SERVER_PATH $TEST_DIR > $tempFile
    
    # Read the output file
    $result = Get-Content -Raw $tempFile
    
    # Delete the temp file
    Remove-Item $tempFile
    
    # Display result
    Write-Host "Response:" -ForegroundColor Yellow
    Write-Host $result
    Write-Host "===========================" -ForegroundColor Gray
    
    return $result
}

Write-Host "`n`n*** STARTING MCP SERVER TESTS ***" -ForegroundColor Green

# Start the server separately to capture startup messages
Start-Process -NoNewWindow -FilePath "node" -ArgumentList "$SERVER_PATH $TEST_DIR" -RedirectStandardOutput "server_start.log" -Wait -WindowStyle Hidden

# 1. Test handshake
Invoke-MCPTest -Method "handshake" -Params @{}

# 2. Test list_tools
Invoke-MCPTest -Method "list_tools" -Params @{}

# 3. Test read_file
Invoke-MCPTest -Method "read_file" -Params @{
    path = "test-file.txt"
}

# 4. Test read_multiple_files
Invoke-MCPTest -Method "read_multiple_files" -Params @{
    paths = @("test-file.txt", "test-script.js", "subdir/nested-file.md")
}

# 5. Test write_file
Invoke-MCPTest -Method "write_file" -Params @{
    path = "write-test.txt"
    content = "This is a file created by the write_file tool test."
}

# 6. Test edit_file
Invoke-MCPTest -Method "edit_file" -Params @{
    path = "test-file.txt"
    edits = @(
        @{
            oldText = "This is a test file for MCP testing."
            newText = "This is a modified test file for MCP testing."
        }
    )
    dryRun = $true
}

# 7. Test create_directory
Invoke-MCPTest -Method "create_directory" -Params @{
    path = "new-test-dir"
}

# 8. Test list_directory
Invoke-MCPTest -Method "list_directory" -Params @{
    path = "."
}

# 9. Test directory_tree
Invoke-MCPTest -Method "directory_tree" -Params @{
    path = "."
}

# 10. Test search_files
Invoke-MCPTest -Method "search_files" -Params @{
    path = "."
    pattern = "*.txt"
}

# 11. Test get_file_info
Invoke-MCPTest -Method "get_file_info" -Params @{
    path = "test-file.txt"
}

# 12. Test list_allowed_directories
Invoke-MCPTest -Method "list_allowed_directories" -Params @{}

Write-Host "`nAll tests completed!" -ForegroundColor Green