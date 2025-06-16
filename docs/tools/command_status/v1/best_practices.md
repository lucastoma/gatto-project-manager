# Command Status Best Practices

## 1. Timing Checks
- Use WaitDurationSeconds=0 for instant status checks
- Set WaitDurationSeconds=5-10 for commands expected to complete soon

## 2. Output Management
- Start with OutputCharacterCount=500 and increase as needed
- For long-running processes, implement output pagination

## 3. Error Handling
- Always check for 'error' field in response
- Handle 'command not found' as a distinct case

## 4. Security
- Never expose full command output with sensitive data
- Validate command IDs against allowlist patterns

## 5. Performance
- Avoid frequent polling (more than once per 5 seconds)
- Cache results when possible for repeated status checks