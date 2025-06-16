# Command Status Error Patterns

## 1. Invalid Command ID
- **Cause**: Command ID doesn't exist or has expired
- **Resolution**: Verify command ID and ensure it's from recent execution

## 2. Premature Status Check
- **Cause**: Checking status before command has started
- **Resolution**: Add brief wait time before checking status

## 3. Output Truncation
- **Cause**: OutputCharacterCount too small for large outputs
- **Resolution**: Increase character count or use paginated approach

## 4. Permission Issues
- **Cause**: Attempting to access commands from other users/sessions
- **Resolution**: Verify command ownership and session context