# Command Status Examples

## Example 1: Checking Status of a Background Process
```json
{
  "CommandId": "cmd_12345",
  "OutputCharacterCount": 500,
  "WaitDurationSeconds": 10
}
```
Checks the status of command `cmd_12345`, waiting up to 10 seconds for completion and retrieving the first 500 characters of output.

## Example 2: Quick Status Check
```json
{
  "CommandId": "cmd_67890",
  "OutputCharacterCount": 100,
  "WaitDurationSeconds": 0
}
```
Immediately checks status without waiting, retrieving the first 100 characters of output if available.