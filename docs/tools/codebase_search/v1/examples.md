# Codebase Search Examples

## Example 1: Basic Function Search
```json
{
  "Query": "parse_user_input",
  "TargetDirectories": [
    "/home/project/src/auth",
    "/home/project/src/utils"
  ]
}
```
Searches for `parse_user_input` function in authentication and utility modules.

## Example 2: Contextual Search
```json
{
  "Query": "user authentication error handling",
  "TargetDirectories": [
    "/home/project/src/auth"
  ]
}
```
Finds code related to authentication error handling using natural language.

## Example 3: Multi-directory Search
```json
{
  "Query": "DatabaseConnection",
  "TargetDirectories": [
    "/home/project/src/db",
    "/home/project/src/api",
    "/home/project/tests"
  ]
}
```
Searches for database connection logic across implementation and test directories.