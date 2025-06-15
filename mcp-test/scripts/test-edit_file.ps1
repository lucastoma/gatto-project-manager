#!/usr/bin/env pwsh
<#!
Purpose: Dedicated regression test script for the `edit_file` tool of the MCP filesystem server.
It covers:
  1. Exact match replacement
  2. Fuzzy match replacement (minor typo)
  3. Overlapping edits to demonstrate validateEdits warning
  4. Dry-run (no write) mode

Usage:  ./scripts/test-edit_file.ps1  [-Verbose]
The script assumes:
  • You have already started the MCP server.
  • `test-tool.ps1` is in the same directory.
!#>

param ([switch]$Verbose)

$ErrorActionPreference = "Stop"
$PSDefaultParameterValues['*:Encoding'] = 'utf8'

$ThisDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Invoker = Join-Path $ThisDir 'test-tool.ps1'
$TempFile = Join-Path $PSScriptRoot 'edit-test.txt'

function Invoke-SafeTool {
    param (
        [string]$ToolName,
        [object]$ToolArguments,
        [string]$Label
    )
    
    $tempFile = [System.IO.Path]::GetTempFileName()
    try {
        $ToolArguments | ConvertTo-Json -Depth 10 | Out-File $tempFile -Encoding utf8
        
        if ($Verbose) { 
            Write-Host "--- $Label ($ToolName) ---" -ForegroundColor Gray
            Write-Host "JSON: $(Get-Content $tempFile -Raw)" -ForegroundColor DarkGray
        }
        
        $output = powershell -Command "& { 
            `$ErrorActionPreference = 'Stop'; 
            . \"$PSScriptRoot\..\test-tool.ps1\" -ToolName '$ToolName' -ParamsJson (Get-Content -Raw -Path '$tempFile')
        }"
        Write-Host $output
    } catch {
        Write-Error "Test '$Label' failed: $_"
        exit 1
    } finally {
        if (Test-Path $tempFile) { Remove-Item $tempFile -Force }
    }
}

# ---------- SETUP ----------
Write-Host "Creating test file $TempFile" -ForegroundColor Cyan
if (Test-Path $TempFile) { Remove-Item $TempFile -Force }
"Initial content plus typo" | Out-File -FilePath $TempFile -Encoding utf8

# 1. Exact match edit
Write-Host "\n[1/4] Testing exact match replacement" -ForegroundColor Yellow
Invoke-SafeTool -ToolName 'edit_file' -Label 'Exact match replacement' -ToolArguments @{
    path  = $TempFile
    edits = @(@{
        oldText = 'Initial content plus typo'
        newText = 'Exact match replacement succeeded'
    })
}

# 2. Fuzzy edit (oldText has a typo -> should still match)
Write-Host "\n[2/4] Testing fuzzy replacement (with typo)" -ForegroundColor Yellow
Invoke-SafeTool -ToolName 'edit_file' -Label 'Fuzzy replacement (typo in oldText)' -ToolArguments @{
    path  = $TempFile
    edits = @(@{
        oldText = 'Exact matc replacement succeeded'  # missing "h"
        newText = 'Fuzzy edit succeeded – typo corrected'
    })
}

# 3. Overlapping edits – triggers validateEdits warning in server log
Write-Host "\n[3/4] Testing overlapping edits (should warn)" -ForegroundColor Yellow
Invoke-SafeTool -ToolName 'edit_file' -Label 'Overlapping edits warning' -ToolArguments @{
    path  = $TempFile
    edits = @(
        @{ oldText = 'Fuzzy edit succeeded – typo corrected'; newText = 'First change' },
        @{ oldText = 'First change'; newText = 'Second change' }
    )
}

# 4. Dry-run – diff only (should not change file)
Write-Host "\n[4/4] Testing dry-run mode" -ForegroundColor Yellow
Invoke-SafeTool -ToolName 'edit_file' -Label 'Dry-run (no write)' -ToolArguments @{
    path   = $TempFile
    dryRun = $true
    edits  = @(@{
        oldText = 'Second change'
        newText = 'This would be written, but dry-run=true'
    })
}

Write-Host "\nAll edit_file scenarios executed. Check:" -ForegroundColor Green
Write-Host "1. Content of $TempFile"
Write-Host "2. Server logs (if LOG_LEVEL=debug)"
