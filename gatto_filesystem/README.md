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
If no directories are specified, it defaults to the current working directory.

**Option 2: Using a configuration file:**

Create a `mcpconfig.json` (or `.yaml`, `.js`) in the project root or specify a path with `--config`.

```bash
npx tsx src/server/index.ts --config /path/to/your/mcpconfig.json
```

Example `mcpconfig.json`:
```json
{
  "allowedDirectories": ["/home/user/projects", "/mnt/shared_data"],
  "logLevel": "info",
  "fileFiltering": {
    "defaultExcludes": [".git", "node_modules", ".DS_Store"],
    "forceTextFiles": false,
    "allowedExtensions": []
  },
  "concurrency": {
    "maxConcurrentRequests": 10,
    "maxConcurrentEdits": 5
  },
  "fuzzyMatching": {
    "minSimilarity": 0.7,
    "maxCandidates": 5,
    "contextWindow": 2,
    "caseSensitive": false,
    "ignoreWhitespace": true
  }
}
```

### Building for Production

```bash
npm run build
```
This creates a `dist` directory with the compiled JavaScript. You can then run `node dist/server/index.js`.

## Configuration

The server can be configured via a configuration file or command-line arguments.

-   `--config <path>`: Path to a JSON, YAML, or JS configuration file.
-   `--log-level <level>`: Logging level (e.g., `trace`, `debug`, `info`, `warn`, `error`, `fatal`).
-   `<allowed_directory_path_1> ...`: Paths to directories the server is allowed to access. If a config file is used, `allowedDirectories` in the file takes precedence. If neither is provided, defaults to the current working directory.

See `src/server/config.ts` and `src/types/config.ts` for all available options and their defaults.

## Available Tools

The server exposes the following tools via MCP `tools/call` requests:

### `read_file`

Reads the content of a file.

**Arguments:**

-   `path` (string, required): Absolute path to the file.
-   `encoding` (string, optional, default: `auto`): Encoding for file content. Can be `utf-8`, `base64`, or `auto` (attempts to detect binary vs. text). If `auto` detects binary, it returns base64.

**Returns:** An object `{ content: string, encoding: 'utf-8' | 'base64', fileType: 'text' | 'binary' | 'unknown' }`.

### `write_file`

Writes content to a file, overwriting it if it exists, or creating it if it doesn't.

**Arguments:**

-   `path` (string, required): Absolute path to the file.
-   `content` (string, required): Content to write.
-   `encoding` (string, optional, default: `utf-8`): Encoding of the provided `content`. Can be `utf-8` or `base64`.

**Returns:** An object `{ success: true }` on success.

### `edit_file`

Applies line-based edits to a text file. Supports fuzzy matching for `oldText`.

**Arguments:**

-   `path` (string, required): Absolute path to the file.
-   `edits` (array of `EditOperation`, required):
    ```typescript
    interface EditOperation {
      oldText: string; // Text to search for (can be slightly inaccurate for fuzzy matching)
      newText: string; // Text to replace with
      forcePartialMatch?: boolean; // If true, allows partial matches above minSimilarity when no exact match is found
    }
    ```
-   `dryRun` (boolean, optional, default: `false`): If `true`, returns a git-style diff instead of applying changes.
-   `debug` (boolean, optional, default: `false`): Show detailed matching information in logs.
-   `caseSensitive` (boolean, optional, default from config): Override config for case sensitivity.
-   `ignoreWhitespace` (boolean, optional, default from config): Override config for ignoring whitespace.
-   `matchConfig` (object, optional): Override fuzzy matching parameters for this specific call.
    ```typescript
    interface FuzzyMatchConfigOverride {
      minSimilarity?: number;
      maxCandidates?: number;
      contextWindow?: number;
    }
    ```

**Returns:** An object `{ diff: string }` if `dryRun` is true, or `{ success: true, operationsAttempted: number, operationsSucceeded: number }` otherwise.

### `list_directory`

Lists the contents of a specified directory.

**Arguments:**

-   `path` (string, required): Absolute path to the directory to list.
-   `recursive` (boolean, optional, default: `false`): If `true`, recursively lists contents of subdirectories. The output will be a flat list of all entries found recursively.
-   `includeFiles` (boolean, optional, default: `true`): Whether to include files in the listing.
-   `includeDirs` (boolean, optional, default: `true`): Whether to include directories in the listing.
-   `includeHidden` (boolean, optional, default: `false`): Whether to include hidden files/directories (those starting with a dot).

**Returns:** An array of `ListDirectoryEntry` objects.

```typescript
interface ListDirectoryEntry {
  name: string;        // Name of the file or directory
  path: string;        // Full absolute path
  type: 'file' | 'directory';
  size?: number;       // Size in bytes (for files)
}
```

### `create_directory`

Creates a new directory. Can create multiple nested directories (like `mkdir -p`).

**Arguments:**

-   `path` (string, required): Absolute path of the directory to create.

**Returns:** An object `{ success: true }` on success.

### `delete_file`

Deletes a file.

**Arguments:**

-   `path` (string, required): Absolute path to the file to delete.

**Returns:** An object `{ success: true }` on success.

### `delete_directory`

Deletes a directory.

**Arguments:**

-   `path` (string, required): Absolute path to the directory to delete.
-   `recursive` (boolean, optional, default: `false`): If `true`, recursively deletes the directory and its contents (like `rm -rf`). Use with caution.

**Returns:** An object `{ success: true }` on success.

### `move_file` (or `rename_file`)

Moves or renames a file or directory.

**Arguments:**

-   `source` (string, required): Absolute path to the source file or directory.
-   `destination` (string, required): Absolute path to the destination.

**Returns:** An object `{ success: true }` on success.

### `get_file_info`

Retrieves metadata about a file or directory.

**Arguments:**

-   `path` (string, required): Absolute path to the file or directory.

**Returns:** An object with file statistics (size, type, timestamps, etc.).
```typescript
interface FileStats {
  name: string;
  path: string;
  type: 'file' | 'directory' | 'symlink' | 'other';
  size: number; // Size in bytes
  createdAt: string; // ISO 8601 timestamp
  modifiedAt: string; // ISO 8601 timestamp
  accessedAt: string; // ISO 8601 timestamp
  isReadable: boolean;
  isWritable: boolean;
  isExecutable: boolean; // For files
  // For directories, may include childrenCount if easily available, otherwise omitted
}
```

### `search_files`

Recursively searches for files and directories matching a pattern.

**Arguments:**

-   `path` (string, required): Root directory to start the search from.
-   `pattern` (string, required): Glob pattern to match (e.g., `*.ts`, `docs/**/*.md`).
-   `excludePatterns` (array of string, optional): Glob patterns to exclude.
-   `useExactPatterns` (boolean, optional, default: `false`): If `true`, uses patterns as provided. If `false` (default), patterns like `*.ts` are treated as `**/*.ts` to search recursively by default.
-   `maxDepth` (number, optional): Maximum depth for recursion.
-   `maxResults` (number, optional): Maximum number of results to return.

**Returns:** An array of strings, where each string is the full absolute path of a matched file or directory.

### `directory_tree`

Gets a recursive tree view of files and directories.

**Arguments:**

-   `path` (string, required): Absolute path to the root directory for the tree.
-   `maxDepth` (number, optional, default: `-1` for unlimited): Maximum depth of the tree.

**Returns:** A `DirectoryTreeEntry` object representing the root of the tree.
```typescript
interface DirectoryTreeEntry {
  name: string;
  path: string;
  type: 'file' | 'directory';
  children?: DirectoryTreeEntry[]; // Undefined for files, array (possibly empty) for directories
}
```

### `list_allowed_directories`

Lists the directories the server is configured to allow access to.

**Arguments:** None.

**Returns:** An array of strings, where each string is an absolute path to an allowed directory.

### `server_stats`

Provides statistics about the server's current state.

**Arguments:** None.

**Returns:** An object containing server statistics like request counts, active file locks, and a snapshot of the current configuration.

## Development

### Setup

```bash
npm install
```

### Running Tests

-   Unit tests: `npm test` or `npm run test:unit`
-   E2E tests: `npm run test:e2e`
-   All tests: `npm run test:all`
-   Coverage: `npm run coverage`

### Linting and Formatting

-   Lint: `npm run lint`
-   Format: `npm run format`

### Building

```bash
npm run build
```

## Protocol Details

The server communicates using JSON-RPC 2.0 messages over stdio. Each message is a JSON string followed by a newline character.

-   **Requests:** Clients send requests with `method`, `params`, and `id`.
-   **Responses:** Server sends responses with `result` (or `error`) and `id`.
-   **Notifications:** Can be sent (e.g., `$/progress`), but this server primarily uses request/response.

Standard MCP methods implemented:
-   `initialize`: Client sends this first. Server may return capabilities.
-   `tools/list`: Client requests available tools. Server returns a list of tool schemas.
-   `tools/call`: Client calls a specific tool with arguments.

## Potential Problems and Areas for Improvement

-   **Error Handling Granularity:** While errors are structured, some internal errors might bubble up as generic "UNKNOWN_ERROR". Continuously refining error codes and messages can improve client-side handling.
-   **Large File Streaming:** For `read_file` and `write_file` with very large files, streaming might be more memory-efficient than reading/writing the entire content at once. This is a common challenge for stdio-based servers.
-   **Configuration Schema Validation:** The server loads config, but a more formal Zod schema for the entire configuration object (`Config` type in `src/types/config.ts`) could be used at startup to validate `mcpconfig.json` more robustly and provide clearer error messages for invalid configurations.
-   **Symlink Handling:** The server currently treats symlinks as 'symlink' type in `get_file_info` but doesn't explicitly resolve them for operations like `read_file`. Behavior for operations on symlinks (e.g., reading a symlink to a file vs. reading the target file) should be clearly documented and consistently implemented. Current `fs.realpath` usage in `validatePath` resolves symlinks for path validation, which is good for security.
-   **`handshake` Method:** The codebase includes a `HandshakeRequestSchema` and handler. The `initialize` method is now the standard MCP way. Consider formally deprecating or removing the `handshake` functionality if it's no longer used or needed.
-   **Root `index.ts` File:** The `index.ts` file at the project root (`/home/lukasz/projects/gatto-ps-ai/index.ts`) is currently empty. This might be intentional or a remnant. If not serving a purpose, it could be removed or documented.
-   **SDK Module Loading Style:** In `src/server/index.ts`, the `@modelcontextprotocol/sdk` modules are loaded using `require`, while other modules use ES6 `import`. This was a specific change made during development. It's worth confirming if this mixed style is the intended long-term approach for consistency.
-   **Fuzzy Matching Documentation:** The `edit_file` tool's fuzzy matching is powerful but complex. More examples in this README or separate documentation illustrating different `matchConfig` scenarios would be beneficial for users.
-   **Performance Considerations:** For tools like `search_files` and `directory_tree` on very large directory structures or with deep recursion, performance could be a concern. Current implementation details (e.g., `maxDepth` limits) help, but further optimization or streaming results might be needed for extreme cases.
-   **Configuration Precedence:** Clearly document the precedence if both command-line arguments (for allowed directories) and a `--config` file are somehow used (though current `getConfig` logic prioritizes `--config` if present, and command-line args are only used for `allowedDirectories` if `--config` is absent).

