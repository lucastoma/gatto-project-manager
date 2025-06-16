# Create Memory Error Patterns

## 1. Missing Required Fields
- **Cause**: Omitting required parameters like Action or CorpusNames
- **Resolution**: Ensure all required fields are provided

## 2. Invalid Memory ID
- **Cause**: Providing an invalid or non-existent ID for update/delete
- **Resolution**: Verify ID exists before update/delete operations

## 3. Duplicate Creation
- **Cause**: Creating duplicate memories for same context
- **Resolution**: Check for existing similar memories before creation

## 4. Permission Issues
- **Cause**: Attempting to modify memories from other workspaces
- **Resolution**: Verify CorpusNames match accessible workspaces