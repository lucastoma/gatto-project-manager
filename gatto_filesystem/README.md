# Gatto Nero MCP Filesystem Server

## Overview

The Gatto Nero MCP Filesystem Server is a Node.js application that implements the Model Context Protocol (MCP) to provide filesystem operations as tools. It allows MCP clients to securely read, write, list, and manage files and directories on the server within configured boundaries.

The server communicates over stdio and uses JSON-RPC for message passing. It features robust path validation, configurable file filtering, fuzzy matching for edits, and detailed logging.

## Features

-   **Secure Access:** Operations are restricted to pre-configured `allowedDirectories`.
-   **Comprehensive Filesystem Tools:**
    -   Read and write files (text and binary with base64 encoding).
    -   List directory contents.
    -   Create, move, and delete files and directories.
    -   Edit text files with line-based replacements and fuzzy matching.
    -   Search for files by pattern.
    -   Retrieve file/directory metadata and tree structures.
-   **Configurable Filtering:** Exclude specific files/directories and enforce text-only file operations.
-   **Concurrency Control:** Manages concurrent edit operations.
-   **Detailed Logging:** Uses Pino for structured logging, including performance metrics.
-   **MCP Compliance:** Implements standard MCP methods like `initialize` and `tools/list`.

## Quick Start

### Prerequisites

-   Node.js (v18 or later recommended)
-   npm (comes with Node.js)

### Installation

1.  Clone the repository (if applicable) or ensure you have the project files.
2.  Navigate to the project root directory (e.g., `/home/lukasz/projects/gatto-ps-ai/`).
3.  Install dependencies:
    ```bash
    npm install
    ```

### Running the Server

The server is started using `tsx` (a TypeScript runner). Commands should be run from the project root (e.g., `/home/lukasz/projects/gatto-ps-ai/`).

**Option 1: Specify allowed directories as arguments:**

```bash
npx tsx src/server/index.ts /path/to/allowed/dir1 /path/to/another/allowed/dir2
```
If no directories are specified, it defaults to the current working directory from where the server is launched.

**Option 2: Using a configuration file:**

Create a `config.json` file (see [Configuration](#configuration) section below for details).
```bash
npx tsx src/server/index.ts --config /path/to/your/config.json
```

The server will then listen for JSON-RPC messages on stdin.

### Basic Usage (Example JSON-RPC)

Once the server is running, an MCP client can send requests.

**Example: Initialize connection**
```json
{
  "jsonrpc": "2.0",
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": { "name": "my-test-client", "version": "1.0.0" }
  },
  "id": 1
}
```

**Example: List available tools**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "params": {},
  "id": 2
}
```
The server will respond on stdout.

## Configuration

The server can be configured using a JSON file passed with the `--config` flag. If not provided, it uses command-line arguments for `allowedDirectories` and defaults for other settings.

**Default `ConfigSchema` structure (see `src/server/config.ts`):**

```typescript
{
  allowedDirectories: string[], // Paths server is allowed to operate within
  fileFiltering: {
    defaultExcludes: string[],  // Glob patterns for files/dirs to exclude
    allowedExtensions: string[],// Glob patterns for allowed file extensions if forceTextFiles is true
    forceTextFiles: boolean     // If true, only allows operations on files with allowedExtensions
  },
  fuzzyMatching: { // For 'edit_file' tool
    maxDistanceRatio: number,   // Max Levenshtein distance / length of oldText
    minSimilarity: number,      // Min similarity for a match
    caseSensitive: boolean,
    ignoreWhitespace: boolean,
    preserveLeadingWhitespace: 'auto' | 'strict' | 'normalize'
  },
  logging: {
    level: 'trace' | 'debug' | 'info' | 'warn' | 'error',
    performance: boolean        // Enable performance logging for operations
  },
  concurrency: {
    maxConcurrentEdits: number, // Max concurrent edits on a single file
    maxGlobalConcurrentEdits: number // Max concurrent edits server-wide
  },
  limits: {
    maxReadBytes: number,       // Max bytes for file reads
    maxWriteBytes: number       // Max bytes for file writes
  }
}
```

**Example `config.json`:**
```json
{
  "allowedDirectories": ["/home/user/projects", "/srv/shared_files"],
  "fileFiltering": {
    "forceTextFiles": true,
    "allowedExtensions": ["*.txt", "*.md", "*.js", "*.json"]
  },
  "logging": {
    "level": "debug",
    "performance": true
  }
}
```

## Available Tools

All tools (except `initialize` and `tools/list`) are invoked using the `tools/call` MCP method.

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "tool_name_here",
    "arguments": {
      // tool-specific arguments
    }
  },
  "id": "request_id_here"
}
```

### Core MCP Methods

#### 1. `initialize`
   - **Description:** Initializes the connection between the client and server, exchanging capabilities and server information.
   - **Request Method:** `initialize`
   - **Arguments (`params`):**
     ```json
     {
       "protocolVersion": "string", // e.g., "2024-11-05"
       "capabilities": {}, // Client capabilities
       "clientInfo": { "name": "string", "version": "string" }
     }
     ```
   - **Response (`result`):**
     ```json
     {
       "protocolVersion": "string",
       "capabilities": { "tools": {} }, // Server capabilities
       "serverInfo": { "name": "mcp-filesystem-server", "version": "string" }
     }
     ```

#### 2. `tools/list`
   - **Description:** Lists all tools provided by the server.
   - **Request Method:** `tools/list`
   - **Arguments (`params`):** `{}` (none)
   - **Response (`result`):**
     ```json
     {
       "tools": [
         { "name": "tool_name", "description": "...", "arguments_schema": { /* JSON Schema */ }, "response_schema": { /* JSON Schema */ } },
         // ... more tools
       ]
     }
     ```
     (Note: `description`, `arguments_schema`, `response_schema` are based on the full tool definition from `src/core/schemas.ts` and `src/core/toolHandlers.ts`)

### Filesystem Tools (invoked via `tools/call`)

For each tool below, the `arguments` object is passed within the `params` of a `tools/call` request.

#### 1. `list_allowed_directories`
   - **Description:** Lists the root directories the server is configured to access.
   - **Arguments:** `{}` (none)
   - **Response (`result.result`):** `{ "directories": ["/path/to/dir1", "/path/to/dir2"] }`

#### 2. `read_file`
   - **Description:** Reads the content of a file.
   - **Arguments:**
     ```json
     {
       "path": "string", // Relative or absolute path to the file
       "encoding": "auto" | "utf-8" | "base64" // (Optional, default: "auto")
     }
     ```
   - **Response (`result.result`):**
     ```json
     {
       "content": "string", // File content (string or base64 encoded string)
       "encodingUsed": "utf-8" | "base64",
       "fileType": "TEXT" | "POTENTIAL_TEXT_WITH_CAVEATS" | "CONFIRMED_BINARY",
       "size": number // File size in bytes
     }
     ```

#### 3. `read_multiple_files`
   - **Description:** Reads content from multiple files.
   - **Arguments:**
     ```json
     {
       "paths": ["string"], // Array of file paths
       "encoding": "auto" | "utf-8" | "base64" // (Optional, default: "auto")
     }
     ```
   - **Response (`result.result`):**
     ```json
     {
       "results": [
         {
           "path": "string",
           "success": boolean,
           "content": "string"?, // Present if success is true
           "encodingUsed": "utf-8" | "base64"?, // Present if success is true
           "error": { "code": "string", "message": "string" }? // Present if success is false
         }
         // ... more results
       ]
     }
     ```

#### 4. `write_file`
   - **Description:** Writes content to a file, creating it if it doesn't exist or overwriting it.
   - **Arguments:**
     ```json
     {
       "path": "string", // Path to the file
       "content": "string", // Content to write (can be base64 for binary)
       "encoding": "utf-8" | "base64" // (Optional, default: "utf-8")
     }
     ```
   - **Response (`result.result`):** `{ "message": "File written successfully." }` (or error)

#### 5. `edit_file`
   - **Description:** Applies a series of edits to a text file. Supports fuzzy matching.
   - **Arguments:**
     ```json
     {
       "path": "string",
       "edits": [
         {
           "oldText": "string", // Text to find and replace
           "newText": "string", // Text to replace with
           "matchConfig": { // Optional, overrides global fuzzy matching config for this edit
             "maxDistanceRatio": number?,
             "minSimilarity": number?,
             "caseSensitive": boolean?,
             "ignoreWhitespace": boolean?,
             "preserveLeadingWhitespace": "auto" | "strict" | "normalize"?,
             "forcePartialMatch": boolean? // If true, attempts partial match even if full match fails
           }?
         }
       ],
       "globalMatchConfig": { /* Same structure as matchConfig, applies to all edits unless overridden */ }?
     }
     ```
   - **Response (`result.result`):**
     ```json
     {
       "diff": "string", // Unified diff of changes
       "matches": [ /* Details about matches and replacements */ ]
     }
     ```

#### 6. `create_directory`
   - **Description:** Creates a new directory (and any necessary parent directories).
   - **Arguments:** `{ "path": "string" }`
   - **Response (`result.result`):** `{ "message": "Directory created successfully." }`

#### 7. `list_directory`
   - **Description:** Lists the contents of a directory.
   - **Arguments:**
     ```json
     {
       "path": "string", // Path to the directory
       "recursive": boolean, // (Optional, default: false) - NOT YET IMPLEMENTED, use directory_tree for recursive
       "includeHidden": boolean // (Optional, default: false) - Filtering based on server config (defaultExcludes)
     }
     ```
   - **Response (`result.result`):**
     ```json
     {
       "entries": [
         { "name": "string", "type": "file" | "directory" | "symlink", "size": number? /* for files */ }
         // ... more entries
       ]
     }
     ```

#### 8. `move_file`
   - **Description:** Moves or renames a file or directory.
   - **Arguments:**
     ```json
     {
       "source": "string",
       "destination": "string"
     }
     ```
   - **Response (`result.result`):** `{ "message": "Moved successfully." }`

#### 9. `delete_file`
   - **Description:** Deletes a file.
   - **Arguments:** `{ "path": "string" }`
   - **Response (`result.result`):** `{ "message": "File deleted successfully." }`

#### 10. `delete_directory`
    - **Description:** Deletes a directory.
    - **Arguments:**
      ```json
      {
        "path": "string",
        "recursive": boolean // (Optional, default: false)
      }
      ```
    - **Response (`result.result`):** `{ "message": "Directory deleted successfully." }`

#### 11. `search_files`
    - **Description:** Searches for files and directories matching a pattern.
    - **Arguments:**
      ```json
      {
        "path": "string", // Starting directory for search
        "pattern": "string", // Glob pattern to match
        "excludePatterns": ["string"]?, // Optional array of glob patterns to exclude
        "maxDepth": number? // Optional maximum depth for recursion
      }
      ```
    - **Response (`result.result`):**
      ```json
      {
        "matches": [
          { "path": "string", "type": "file" | "directory" }
          // ... more matches
        ]
      }
      ```

#### 12. `get_file_info`
    - **Description:** Retrieves metadata for a file or directory.
    - **Arguments:** `{ "path": "string" }`
    - **Response (`result.result`):**
      ```json
      {
        "path": "string",
        "type": "file" | "directory" | "symlink" | "other",
        "size": number, // In bytes
        "created": "ISO8601_string", // Creation timestamp
        "modified": "ISO8601_string", // Last modified timestamp
        "accessed": "ISO8601_string", // Last accessed timestamp
        "permissions": "string" // e.g., "rw-r--r--"
      }
      ```

#### 13. `directory_tree`
    - **Description:** Gets a recursive tree view of files and directories.
    - **Arguments:**
      ```json
      {
        "path": "string",
        "maxDepth": number? // Optional, default: unlimited (-1)
      }
      ```
    - **Response (`result.result`):**
      ```json
      {
        "tree": { // DirectoryTreeEntry structure
          "name": "string",
          "path": "string",
          "type": "directory",
          "children": [ /* Array of DirectoryTreeEntry for files/subdirs */ ]
        }
      }
      // DirectoryTreeEntry for a file: { "name": "string", "path": "string", "type": "file", "size": number }
      ```

#### 14. `server_stats`
    - **Description:** Retrieves statistics about the server's operation.
    - **Arguments:** `{}` (none)
    - **Response (`result.result`):**
      ```json
      {
        "requestCount": number,
        "editOperationCount": number,
        "activeLocks": number,
        "config": { /* Current server configuration */ }
      }
      ```

## Error Handling

The server returns structured errors as per MCP guidelines. Errors include a `code`, `message`, and often `hint` and `details`.
Refer to `src/types/errors.ts` and `src/utils/hintMap.ts` for common error codes and hints.

Example error response:
```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32000, // Generic server error code range for MCP
    "message": "Tool execution failed",
    "data": { // This is the StructuredError
      "code": "ACCESS_DENIED",
      "message": "Path '/forbidden/file.txt' is outside allowed directories.",
      "hint": "Path is outside of allowedDirectories. Check the 'path' argument.",
      "details": { /* ... */ }
    }
  },
  "id": "request_id"
}
```

## Logging

The server uses [Pino](https://getpino.io/) for logging.
- Logs are output to `stderr`.
- Logs are also written to `logs/mcp-filesystem.log` in the directory where the server is started (info level and above).
- Log level can be configured (see [Configuration](#configuration)).
- Performance logs can be enabled for debugging.

## Development

### Running Tests

End-to-end tests are located in `src/e2e/__tests__`.
To run tests (from the project root, e.g., `/home/lukasz/projects/gatto-ps-ai/`):
```bash
npm run test:e2e
```
This command uses Jest. The `package.json` includes:
`"test:e2e": "jest --config jest.e2e.config.js --detectOpenHandles"`

## Potential Issues & Areas for Improvement

-   **Legacy `handshake` Method:** The `src/core/toolHandlers.ts` includes a `HandshakeRequestSchema` and handler. The `initialize` method is now the standard MCP way. Consider formally deprecating or removing the `handshake` functionality if it's no longer used or needed.
-   **Root `index.ts` File:** The `index.ts` file at the project root (`/home/lukasz/projects/gatto-ps-ai/index.ts`) is currently empty. This might be intentional or a remnant. If not serving a purpose, it could be removed or documented.
-   **SDK Module Loading Style:** In `src/server/index.ts`, the `@modelcontextprotocol/sdk` modules are loaded using `require`, while other modules use ES6 `import`. This was a specific change made during development. It's worth confirming if this mixed style is the intended long-term approach for consistency.
-   **`list_directory` Recursion:** The `list_directory` tool has a `recursive` parameter in its schema, but the implementation notes it's not yet implemented and suggests using `directory_tree`. This could be clarified by removing the `recursive` parameter from `list_directory`'s schema if `directory_tree` is the sole method for recursive listing, or by implementing the recursive functionality in `list_directory`.
-   **Fuzzy Matching Documentation:** The `edit_file` tool's fuzzy matching is powerful but complex. More examples in this README or separate documentation illustrating different `matchConfig` scenarios would be beneficial for users.
-   **Performance Considerations:** For tools like `search_files` and `directory_tree` on very large directory structures or with deep recursion, performance could be a concern. Current implementation details (e.g., `maxDepth` limits) help, but further optimization or streaming results might be needed for extreme cases.
-   **Configuration Precedence:** Clearly document the precedence if both command-line arguments (for allowed directories) and a `--config` file are somehow used (though current `getConfig` logic prioritizes `--config` if present, and command-line args are only used for `allowedDirectories` if `--config` is absent).
