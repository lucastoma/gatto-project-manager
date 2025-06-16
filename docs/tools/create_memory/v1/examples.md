# Create Memory Examples

## Example 1: Creating a New Memory
```json
{
  "Id": "",
  "Title": "User Preference: Dark Mode",
  "Content": "The user prefers dark mode in all applications",
  "CorpusNames": ["lucastoma/GattoNeroPhotoshop"],
  "Tags": ["preference", "ui"],
  "Action": "create",
  "UserTriggered": false
}
```
Creates a new memory about user's UI preference.

## Example 2: Updating an Existing Memory
```json
{
  "Id": "memory_12345",
  "Title": "User Preference: Dark Mode",
  "Content": "The user prefers dark mode and has requested larger fonts",
  "CorpusNames": ["lucastoma/GattoNeroPhotoshop"],
  "Tags": ["preference", "ui", "accessibility"],
  "Action": "update",
  "UserTriggered": false
}
```
Updates an existing memory with additional information.

## Example 3: Deleting a Memory
```json
{
  "Id": "memory_67890",
  "Title": "",
  "Content": "",
  "CorpusNames": ["lucastoma/GattoNeroPhotoshop"],
  "Tags": [],
  "Action": "delete",
  "UserTriggered": false
}
```
Deletes an existing memory by ID.