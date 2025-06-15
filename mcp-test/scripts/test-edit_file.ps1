#!/usr/bin/env pwsh
<#!
Purpose: Dedicated regression test script for the `edit_file` tool of the MCP filesystem server.
It covers:
  1. Exact match replacement
  2. Fuzzy match replacement (minor typo)
  3. Overlapping edits to demonstrate validateEdits warning
  4. Dry-run (no write) mode
  5. Partial Match (No Force) - Expect Error with Details
  6. Partial Match (Force Accept) - Expect Success
  7. Fuzzy Match (Force Accept, Low Similarity) - Expect Success
  8. No Match (Force Accept, Below Threshold) - Expect FUZZY_MATCH_FAILED
  9. Excluded directory - Expect FILE_FILTERED_OUT
  10. Disallowed extension - Expect FILE_FILTERED_OUT
  11. Binary file detection - Expect BINARY_FILE_ERROR

Usage:  ./scripts/test-edit_file.ps1  [-Verbose]
The script assumes:
  • You have already started the MCP server.
  • `test-tool.ps1` is in the same directory.
!#>

param ([switch]$Verbose)

$ErrorActionPreference = "Stop"
$PSDefaultParameterValues['*:Encoding'] = 'utf8'

$ThisDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Invoker = Join-Path $ThisDir '..\test-tool.ps1' # Corrected path
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
        
        $output = pwsh -Command "& { 
            `$ErrorActionPreference = 'Stop'; 
            . '$PSScriptRoot/../test-tool.ps1' -ToolName '$ToolName' -ParamsJson (Get-Content -Raw -Path '$tempFile')
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
@"
Line 1: This is the first line of the original content.
Line 2: Here is the second line, which we will try to partially match.
Line 3: A third line for context and other operations.
Line 4: Another line for good measure.
Line 5: The final line before edits begin.
"@ | Out-File -FilePath $TempFile -Encoding utf8

# 1. Exact match edit
Write-Host "\n[1/11] Testing exact match replacement" -ForegroundColor Yellow
Invoke-SafeTool -ToolName 'edit_file' -Label 'Exact match replacement' -ToolArguments @{
    path  = $TempFile
    edits = @(@{
        oldText = 'Line 1: This is the first line of the original content.'
        newText = 'Line 1: Exact match replacement succeeded.'
    })
}

# 2. Fuzzy edit (oldText has a typo -> should still match)
Write-Host "\n[2/11] Testing fuzzy replacement (with typo)" -ForegroundColor Yellow
Invoke-SafeTool -ToolName 'edit_file' -Label 'Fuzzy replacement (typo in oldText)' -ToolArguments @{
    path  = $TempFile
    edits = @(@{
        oldText = 'Line 1: Exact matc replacement succeeded.'  # missing "h"
        newText = 'Line 1: Fuzzy edit succeeded – typo corrected.'
    })
}

# 3. Overlapping edits – triggers validateEdits warning in server log
Write-Host "\n[3/11] Testing overlapping edits (should warn)" -ForegroundColor Yellow
Invoke-SafeTool -ToolName 'edit_file' -Label 'Overlapping edits warning' -ToolArguments @{
    path  = $TempFile
    edits = @(
        @{ oldText = 'Line 1: Fuzzy edit succeeded – typo corrected.'; newText = 'Line 1: First change for overlapping test.' },
        @{ oldText = 'Line 1: First change for overlapping test.'; newText = 'Line 1: Second change, overlap complete.' }
    )
}

# 4. Dry-run – diff only (should not change file)
Write-Host "\n[4/11] Testing dry-run mode" -ForegroundColor Yellow
Invoke-SafeTool -ToolName 'edit_file' -Label 'Dry-run (no write)' -ToolArguments @{
    path   = $TempFile
    dryRun = $true
    edits  = @(@{
        oldText = 'Line 1: Second change, overlap complete.'
        newText = 'Line 1: This would be written, but dry-run=true'
    })
}

# ---------- NEW TESTS FOR forcePartialMatch ----------

# 5. Partial Match (No Force) - Expect Error with Details
Write-Host "\n[5/11] Testing Partial Match (No Force) - Expect Error" -ForegroundColor Yellow
Invoke-SafeTool -ToolName 'edit_file' -Label 'Partial Match (No Force)' -ToolArguments @{
    path  = $TempFile
    edits = @(@{
        oldText = 'second line, which we will try'
        newText = 'This should not be written due to partial match error'
        # forcePartialMatch = $false (default)
    })
} # Expect PARTIAL_MATCH error with details

# 6. Partial Match (Force Accept) - Expect Success
Write-Host "\n[6/11] Testing Partial Match (Force Accept) - Expect Success" -ForegroundColor Yellow
Invoke-SafeTool -ToolName 'edit_file' -Label 'Partial Match (Force Accept)' -ToolArguments @{
    path  = $TempFile
    edits = @(@{
        oldText = 'second line, which we will try' # Same partial as above
        newText = 'Line 2: Partial match successfully forced and applied.'
        forcePartialMatch = $true
    })
}

# 7. Fuzzy Match (Force Accept, Low Similarity but meets minSimilarity) - Expect Success
# Assumes minSimilarity in fuzzyEdit.ts is around 0.5-0.6, and this match is slightly above it.
Write-Host "\n[7/11] Testing Fuzzy Match (Force Accept, Low Similarity) - Expect Success" -ForegroundColor Yellow
Invoke-SafeTool -ToolName 'edit_file' -Label 'Fuzzy Match (Force Accept, Low Similarity)' -ToolArguments @{
    path  = $TempFile
    edits = @(@{
        oldText = 'A thrd lne fr cntxt nd othr oprtns.' # Very fuzzy match for "Line 3: A third line for context and other operations."
        newText = 'Line 3: Fuzzy forced match (low similarity) applied.'
        forcePartialMatch = $true
    })
}

# 8. No Match (Force Accept, Below Threshold) - Expect FUZZY_MATCH_FAILED
# This oldText is intentionally very different to be below any reasonable minSimilarity
Write-Host "\n[8/11] Testing No Match (Force Accept, Below Threshold) - Expect FUZZY_MATCH_FAILED" -ForegroundColor Yellow
Invoke-SafeTool -ToolName 'edit_file' -Label 'No Match (Force Accept, Below Threshold)' -ToolArguments @{
    path  = $TempFile
    edits = @(@{
        oldText = 'Completely unrelated gibberish text that will not match anything.'
        newText = 'This text should never be written.'
        forcePartialMatch = $true
    })
} # Expect FUZZY_MATCH_FAILED error

# 9. Excluded directory - Expect FILE_FILTERED_OUT
Write-Host "\n[9/11] Testing excluded directory (node_modules) - Expect FILE_FILTERED_OUT" -ForegroundColor Cyan
$excludedDirTestFile = Join-Path $PSScriptRoot '../node_modules/excluded-test.txt'
New-Item -Path $excludedDirTestFile -ItemType File -Force | Out-Null
Invoke-SafeTool -ToolName 'edit_file' -Label 'Excluded Directory' -ToolArguments @{
    path  = $excludedDirTestFile
    edits = @(@{
        oldText = 'test'
        newText = 'should fail'
    })
}

# 10. Disallowed extension - Expect FILE_FILTERED_OUT
Write-Host "\n[10/11] Testing disallowed extension (.jpg) - Expect FILE_FILTERED_OUT" -ForegroundColor Cyan
$imageTestFile = Join-Path $PSScriptRoot 'test.jpg'
New-Item -Path $imageTestFile -ItemType File -Force | Out-Null
Invoke-SafeTool -ToolName 'edit_file' -Label 'Disallowed Extension' -ToolArguments @{
    path  = $imageTestFile
    edits = @(@{
        oldText = 'test'
        newText = 'should fail'
    })
}

# 11. Binary file detection - Expect BINARY_FILE_ERROR
Write-Host "\n[11/11] Testing binary file detection - Expect BINARY_FILE_ERROR" -ForegroundColor Cyan
$binaryTestFile = Join-Path $PSScriptRoot 'binary-test.bin'
[byte[]]$bytes = 0x00,0x01,0x02,0x03
[System.IO.File]::WriteAllBytes($binaryTestFile, $bytes)
Invoke-SafeTool -ToolName 'edit_file' -Label 'Binary File' -ToolArguments @{
    path  = $binaryTestFile
    edits = @(@{
        oldText = 'test'
        newText = 'should fail'
    })
}

Write-Host "\nAll 11 edit_file scenarios executed. Check:" -ForegroundColor Green
Write-Host "1. Content of $TempFile"
Write-Host "2. Server logs (if LOG_LEVEL=debug)"
Write-Host "3. Console output for expected errors"
