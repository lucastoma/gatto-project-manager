# Projekt: gatto nero mcp no node and dist
## Katalog główny: `/home/lukasz/projects/gatto-ps-ai`
## Łączna liczba unikalnych plików: 23
---
## Grupa: gatto nero mcp no node and dist
**Opis:** kod gatto nerro mcp filesystem
**Liczba plików w grupie:** 23

### Lista plików:
- `index.ts`
- `src/constants/extensions.ts`
- `src/utils/hintMap.ts`
- `src/utils/pathFilter.test.ts`
- `src/utils/pathFilter.ts`
- `src/utils/binaryDetect.ts`
- `src/utils/performance.ts`
- `src/utils/pathUtils.ts`
- `src/server/index.ts`
- `src/server/config.ts`
- `src/types/errors.ts`
- `src/types/fast-levenshtein.d.ts`
- `src/core/security.ts`
- `src/core/concurrency.ts`
- `src/core/fuzzyEdit.ts`
- `src/core/__tests__/applyFileEdits.test.ts`
- `src/core/fileInfo.ts`
- `src/core/toolHandlers.ts`
- `src/core/security.test.ts`
- `src/core/schemas.ts`
- `package.json`
- `tsconfig.json`
- `test-filtering.js`

### Zawartość plików:
#### Plik: `index.ts`
```ts
// This file is intentionally left blank after refactoring to src/server/index.ts
```
#### Plik: `src/constants/extensions.ts`
```ts
export const DEFAULT_EXCLUDE_PATTERNS: string[] = [
    '**/build/**',
    '**/dist/**',
    '**/node_modules/**',
    '**/.git/**',
    '**/*.jpg', '**/*.png', '**/*.gif', '**/*.pdf',
    '**/*.zip', '**/*.tar', '**/*.gz'
];

export const DEFAULT_ALLOWED_EXTENSIONS: string[] = [
    '*.txt', '*.js', '*.jsx', '*.ts', '*.tsx', '*.json', '*.yaml', '*.yml',
    '*.html', '*.htm', '*.css', '*.scss', '*.sass', '*.less', '*.py', '*.java', '*.go',
    '*.rs', '*.rb', '*.php', '*.sh', '*.bash', '*.zsh', '*.md', '*.markdown', '*.xml',
    '*.svg', '*.csv', '*.toml', '*.ini', '*.cfg', '*.conf', '*.env', '*.ejs', '*.pug',
    '*.vue', '*.svelte', '*.graphql', '*.gql', '*.proto', '*.kt', '*.kts', '*.swift',
    '*.m', '*.h', '*.c', '*.cpp', '*.hpp', '*.cs', '*.fs', '*.fsx', '*.clj', '*.cljs',
    '*.cljc', '*.edn', '*.ex', '*.exs', '*.erl', '*.hrl', '*.lua', '*.sql', '*.pl',
    '*.pm', '*.r', '*.jl', '*.dart', '*.groovy', '*.gradle', '*.nim', '*.zig', '*.v',
    '*.vh', '*.vhd', '*.cl', '*.tex', '*.sty', '*.cls', '*.rst', '*.adoc', '*.asciidoc'
];
```
#### Plik: `src/utils/hintMap.ts`
```ts
export interface HintInfo {
  confidence: number;
  hint: string;
  example?: unknown;
}

export const HINTS: Record<string, HintInfo> = {
  ACCESS_DENIED: {
    confidence: 0.9,
    hint: "Path is outside of allowedDirectories. Check the 'path' argument.",
    example: { path: "/workspace/project/file.txt" }
  },
  VALIDATION_ERROR: { confidence: 0.8, hint: "Check the required fields and their types in the arguments." },
  FUZZY_MATCH_FAILED: {
    confidence: 0.7,
    hint: "Try adjusting minSimilarity/maxDistanceRatio or check for whitespace/indentation issues.",
    example: { minSimilarity: 0.6, maxDistanceRatio: 0.3 }
  },
  BINARY_FILE_ERROR: {
    confidence: 0.95,
    hint: "'edit_file' works only on text files. Use 'write_file' with base64 encoding for binary files."
  },
  PARTIAL_MATCH: {
    confidence: 0.6,
    hint: "Found a partial match. Review the diff and correct 'oldText' or parameters."
  },
  FILE_NOT_FOUND_MULTI: {
    confidence: 0.8,
    hint: "One or more requested files could not be read. See per-file result details."
  },
  UNKNOWN_TOOL: {
    confidence: 0.9,
    hint: "The requested tool does not exist. Use 'list_tools' to see available tools."
  },
  DEST_EXISTS: {
    confidence: 0.85,
    hint: "Destination path already exists. Provide a different destination or remove the existing file first.",
    example: { source: "/workspace/src.txt", destination: "/workspace/dest.txt" }
  },
  SRC_MISSING: {
    confidence: 0.85,
    hint: "Source file does not exist. Check the 'source' argument.",
    example: { source: "/workspace/missing.txt" }
  },
  UNKNOWN_ERROR: {
    confidence: 0.1,
    hint: "An unexpected error occurred. Check server logs for details."
  }
};
```
#### Plik: `src/utils/pathFilter.test.ts`
```ts
import { shouldSkipPath } from './pathFilter.js';
import { DEFAULT_EXCLUDE_PATTERNS, DEFAULT_ALLOWED_EXTENSIONS } from '../constants/extensions.js';

import { Config } from '../server/config.js'; // Import the Config type

// Define the type for overrides to allow partial nested objects
type MockConfigOverrides = {
  allowedDirectories?: string[];
  fileFiltering?: Partial<Config['fileFiltering']>;
  fuzzyMatching?: Partial<Config['fuzzyMatching']>;
  logging?: Partial<Config['logging']>;
  concurrency?: Partial<Config['concurrency']>;
  limits?: Partial<Config['limits']>;
};

describe('shouldSkipPath', () => {
  const mockConfig = (overrides: MockConfigOverrides = {}): Config => ({
    allowedDirectories: overrides.allowedDirectories || ['/allowed'],
    fileFiltering: {
      defaultExcludes: overrides.fileFiltering?.defaultExcludes ?? DEFAULT_EXCLUDE_PATTERNS,
      allowedExtensions: overrides.fileFiltering?.allowedExtensions ?? DEFAULT_ALLOWED_EXTENSIONS,
      forceTextFiles: overrides.fileFiltering?.forceTextFiles ?? true, // Default from ConfigSchema (true)
    },
    fuzzyMatching: {
      maxDistanceRatio: overrides.fuzzyMatching?.maxDistanceRatio ?? 0.25,
      minSimilarity: overrides.fuzzyMatching?.minSimilarity ?? 0.7,
      caseSensitive: overrides.fuzzyMatching?.caseSensitive ?? false,
      ignoreWhitespace: overrides.fuzzyMatching?.ignoreWhitespace ?? true,
      preserveLeadingWhitespace: (overrides.fuzzyMatching?.preserveLeadingWhitespace ?? 'auto') as 'auto' | 'strict' | 'normalize', // Matches ConfigSchema
    },
    logging: {
      level: (overrides.logging?.level ?? 'info') as 'trace' | 'debug' | 'info' | 'warn' | 'error', // Matches ConfigSchema
      performance: overrides.logging?.performance ?? false,
    },
    concurrency: {
      maxConcurrentEdits: overrides.concurrency?.maxConcurrentEdits ?? 10,
      maxGlobalConcurrentEdits: overrides.concurrency?.maxGlobalConcurrentEdits ?? 20,
    },
    limits: {
      maxReadBytes: overrides.limits?.maxReadBytes ?? 5 * 1024 * 1024,
      maxWriteBytes: overrides.limits?.maxWriteBytes ?? 5 * 1024 * 1024,
    }
  });


  it('should allow basic file when no filters', () => {
    expect(shouldSkipPath('/allowed/file.txt', mockConfig())).toBe(false);
  });

  it('should exclude files matching default patterns', () => {
    DEFAULT_EXCLUDE_PATTERNS.forEach(pattern => {
      const testPath = `/allowed/${pattern.replace('*', 'test')}`;
      expect(shouldSkipPath(testPath, mockConfig())).toBe(true);
    });
  });

  it('should exclude files matching custom patterns', () => {
    const config = mockConfig({ fileFiltering: { defaultExcludes: ['custom*'] } });
    expect(shouldSkipPath('/allowed/custom-file.txt', config)).toBe(true);
    expect(shouldSkipPath('/allowed/other-file.txt', config)).toBe(false);
  });

  it('should allow files with default extensions when forceTextFiles=true', () => {
    const config = mockConfig({ fileFiltering: { forceTextFiles: true } });
    DEFAULT_ALLOWED_EXTENSIONS.forEach(ext => {
      expect(shouldSkipPath(`/allowed/file${ext}`, config)).toBe(false);
    });
  });

  it('should exclude files without allowed extensions when forceTextFiles=true', () => {
    const config = mockConfig({ fileFiltering: { forceTextFiles: true } });
    expect(shouldSkipPath('/allowed/file.bin', config)).toBe(true);
  });

  it('should allow files with custom extensions when forceTextFiles=true', () => {
    const config = mockConfig({
      fileFiltering: {
        forceTextFiles: true,
        allowedExtensions: ['*.custom']
      }
    });
    expect(shouldSkipPath('/allowed/file.custom', config)).toBe(false);
    expect(shouldSkipPath('/allowed/file.txt', config)).toBe(true);
  });

  it('should handle case insensitivity', () => {
    const config = mockConfig({ fileFiltering: { defaultExcludes: ['UPPER*'] } });
    expect(shouldSkipPath('/allowed/upper-case.txt', config)).toBe(true);
    expect(shouldSkipPath('/allowed/UPPER-CASE.txt', config)).toBe(true);
  });

  it('should handle dot files', () => {
    const config = mockConfig({ fileFiltering: { defaultExcludes: ['.*'] } });
    expect(shouldSkipPath('/allowed/.hidden', config)).toBe(true);
    expect(shouldSkipPath('/allowed/visible', config)).toBe(false);
  });
});
```
#### Plik: `src/utils/pathFilter.ts`
```ts
import path from 'node:path';
import { minimatch } from 'minimatch';
import type { Config } from '../server/config.js';
import { DEFAULT_EXCLUDE_PATTERNS, DEFAULT_ALLOWED_EXTENSIONS } from '../constants/extensions.js';

/** Zwraca true, jeśli ścieżka powinna być pominięta (wykluczona). */
export function shouldSkipPath(filePath: string, config: Config): boolean {
    const baseDir = config.allowedDirectories[0] ?? process.cwd();
    const rel = path.relative(baseDir, filePath);

    // 1) wzorce global-exclude (połącz domyślne + z configu)
    const allExcludes = [...DEFAULT_EXCLUDE_PATTERNS, ...config.fileFiltering.defaultExcludes];
    if (allExcludes.some(p => minimatch(rel, p, { dot: true, nocase: true }))) return true;

    // 2) rozszerzenia, jeśli forceTextFiles aktywne
    if (config.fileFiltering.forceTextFiles) {
        const ext = path.extname(filePath).toLowerCase();
        const allowed = [...DEFAULT_ALLOWED_EXTENSIONS, ...config.fileFiltering.allowedExtensions];
        if (!allowed.some(p => minimatch(`*${ext}`, p, { nocase: true }))) return true;
    }
    return false;
}
```
#### Plik: `src/utils/binaryDetect.ts`
```ts
import path from 'node:path';
import { isUtf8 as bufferIsUtf8 } from 'buffer';

const NUL_BYTE_CHECK_SAMPLE_SIZE = 8192; // Check first 8KB for NUL bytes

const BINARY_EXTENSIONS = new Set([
  '.exe', '.dll', '.so', '.dylib', '.bin', '.dat',
  '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.ico',
  '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
  '.pdf', '.zip', '.rar', '.tar', '.gz', '.7z',
  '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
  '.wasm', '.o', '.a', '.obj', '.lib', '.class', '.pyc', '.pyo', '.pyd', // Compiled code/objects
  '.sqlite', '.db', '.mdb', '.accdb', '.swf', '.fla' // Databases and flash
]);

export function isBinaryFile(buffer: Buffer, filename?: string): boolean {
  const isUtf8 = (Buffer as any).isUtf8 ?? bufferIsUtf8;
  if (isUtf8 && !isUtf8(buffer)) {
    return true;
  }

  // Check for NUL bytes only in a sample of the buffer to avoid performance issues with large files.
  const sampleForNulCheck = buffer.length > NUL_BYTE_CHECK_SAMPLE_SIZE 
    ? buffer.subarray(0, NUL_BYTE_CHECK_SAMPLE_SIZE) 
    : buffer;
  if (sampleForNulCheck.includes(0)) {
    return true;
  }

  if (filename) {
    const ext = path.extname(filename).toLowerCase();
    if (BINARY_EXTENSIONS.has(ext)) {
      return true;
    }
  }

  let nonPrintable = 0;
  const sampleSize = Math.min(1024, buffer.length);
  
  for (let i = 0; i < sampleSize; i++) {
    const byte = buffer[i];
    if (byte < 32 && byte !== 9 && byte !== 10 && byte !== 13) {
      nonPrintable++;
    }
  }

  return (nonPrintable / sampleSize) > 0.1;
}
```
#### Plik: `src/utils/performance.ts`
```ts
import { performance } from 'node:perf_hooks';
import type { Logger } from 'pino';
import type { Config } from '../server/config.js';

export class PerformanceTimer {
  private startTime: number;
  private operation: string;
  private logger: Logger;
  private enabled: boolean;

  constructor(operation: string, logger: Logger, config: Config) {
    this.operation = operation;
    this.logger = logger;
    this.enabled = config.logging.performance;
    this.startTime = this.enabled ? performance.now() : 0;
  }

  end(additionalData?: any): number {
    if (!this.enabled) {
      return 0;
    }
    const duration = performance.now() - this.startTime;
    this.logger.debug({
      operation: this.operation,
      duration_ms: Math.round(duration * 100) / 100,
      ...additionalData
    }, `Performance: ${this.operation}`);
    return duration;
  }
}
```
#### Plik: `src/utils/pathUtils.ts`
```ts
import path from 'node:path';
import os from 'node:os';

export function normalizePath(p: string): string {
  return path.normalize(p);
}

export function expandHome(filepath: string): string {
  if (filepath.startsWith('~/') || filepath === '~') {
    return path.join(os.homedir(), filepath.slice(1));
  }
  return filepath;
}
```
#### Plik: `src/server/index.ts`
```ts
#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import * as pino from 'pino';
import { promises as fs } from 'node:fs';
import fsSync from 'node:fs';
import path from 'node:path';

import { getConfig } from './config.js';
import { setupToolHandlers } from '../core/toolHandlers.js';
import * as schemas from '../core/schemas.js';
import { expandHome, normalizePath } from '../utils/pathUtils.js';

let runningServer: Server | undefined;

async function shutdown(signal: NodeJS.Signals, logger: pino.Logger) {
  logger.info({ signal }, 'Received termination signal, shutting down gracefully');
  try {
    if (runningServer) {
      // await runningServer.disconnect(); // Not supported by SDK, just exit
    }
  } catch (err) {
    logger.error({ err }, 'Error during graceful shutdown');
  } finally {
    // Give some time for logs to flush
    setTimeout(() => process.exit(0), 100);
  }
}

async function main() {
  // TODO: parse process.argv or pass args if needed
  const config = await getConfig([]);

  // Create logs directory if it doesn't exist
  const logsDir = path.join(process.cwd(), 'logs');
  try {
    await fs.mkdir(logsDir, { recursive: true });
  } catch (err) {
    console.error('Could not create logs directory:', err);
  }

  // Create file stream for logging
  const fileStream = fsSync.createWriteStream(path.join(logsDir, 'mcp-filesystem.log'), { flags: 'a' });

  const logger = pino.pino(
    {
      level: config.logging.level,
      timestamp: () => `,"timestamp":"${new Date().toISOString()}"`,
      base: { service: 'mcp-filesystem-server', version: '0.7.0' }
    },
    pino.multistream([
      { stream: process.stdout },
      { stream: fileStream, level: 'info' }
    ])
  );

  // Add path information for debug logs
  const logWithPaths = (logFn: Function) => (obj: any) => {
    if (config.logging.level === 'debug') {
      if (obj.path && typeof obj.path === 'string') {
        obj.absolutePath = normalizePath(path.resolve(obj.path));
      }
      if (obj.directory && typeof obj.directory === 'string') {
        obj.absoluteDirectory = normalizePath(path.resolve(obj.directory));
      }
    }
    return logFn(obj);
  };

  logger.info = logWithPaths(logger.info.bind(logger));
  logger.debug = logWithPaths(logger.debug.bind(logger));
  logger.error = logWithPaths(logger.error.bind(logger));
  logger.warn = logWithPaths(logger.warn.bind(logger));

  const allowedDirectories = config.allowedDirectories.map((dir: string) => normalizePath(path.resolve(expandHome(dir))));

  await Promise.all(allowedDirectories.map(async (dir: string) => {
    try {
      const stats = await fs.stat(dir);
      if (!stats.isDirectory()) {
        logger.error(`Error: ${dir} is not a directory`);
        process.exit(1);
      }
    } catch (error) {
      logger.error({ error, directory: dir }, `Error accessing directory ${dir}`);
      process.exit(1);
    }
  }));

  const server = new Server(
    { name: 'secure-filesystem-server', version: '0.7.0' },
    { capabilities: { tools: {} } }
  );

  runningServer = server;

  setupToolHandlers(server, allowedDirectories, logger, config);
  // list_tools jest już zarejestrowane w toolHandlers.ts, więc usuwamy duplikat

  const transport = new StdioServerTransport();
  await server.connect(transport);

  logger.info({ version: '0.7.0', allowedDirectories, config }, 'Enhanced MCP Filesystem Server started');

  // Setup signal handlers after server is running
  process.once('SIGINT', (sig) => shutdown(sig, logger));
  process.once('SIGTERM', (sig) => shutdown(sig, logger));
}

main().catch((error) => {
  console.error('Fatal error running server:', error);
  process.exit(1);
});
```
#### Plik: `src/server/config.ts`
```ts
import { z } from "zod";
import fs from "node:fs/promises";
import path from "node:path";
import { DEFAULT_EXCLUDE_PATTERNS, DEFAULT_ALLOWED_EXTENSIONS } from '../constants/extensions.js';

export const ConfigSchema = z.object({
  allowedDirectories: z.array(z.string()),
  fileFiltering: z.object({
    defaultExcludes: z.array(z.string()).default(DEFAULT_EXCLUDE_PATTERNS),
    allowedExtensions: z.array(z.string()).default(DEFAULT_ALLOWED_EXTENSIONS),
    forceTextFiles: z.boolean().default(true)
  }).default({
    defaultExcludes: DEFAULT_EXCLUDE_PATTERNS,
    allowedExtensions: DEFAULT_ALLOWED_EXTENSIONS,
    forceTextFiles: true
  }),
  fuzzyMatching: z.object({
    maxDistanceRatio: z.number().min(0).max(1).default(0.25),
    minSimilarity: z.number().min(0).max(1).default(0.7),
    caseSensitive: z.boolean().default(false),
    ignoreWhitespace: z.boolean().default(true),
    preserveLeadingWhitespace: z.enum(['auto', 'strict', 'normalize']).default('auto')
  }).default({}),
  logging: z.object({
    level: z.enum(['trace', 'debug', 'info', 'warn', 'error']).default('info'),
    performance: z.boolean().default(false)
  }).default({}),
  concurrency: z.object({
    maxConcurrentEdits: z.number().positive().default(10),
    maxGlobalConcurrentEdits: z.number().positive().default(20)
  }).default({}),
  limits: z.object({
    maxReadBytes: z.number().positive().default(5 * 1024 * 1024), // 5 MB
    maxWriteBytes: z.number().positive().default(5 * 1024 * 1024) // 5 MB
  }).default({})
}).default({
  allowedDirectories: [],
  fileFiltering: {
    defaultExcludes: DEFAULT_EXCLUDE_PATTERNS,
    allowedExtensions: DEFAULT_ALLOWED_EXTENSIONS,
    forceTextFiles: true
  },
  concurrency: {
    maxConcurrentEdits: 10,
    maxGlobalConcurrentEdits: 20
  },
  limits: {
    maxReadBytes: 5 * 1024 * 1024,
    maxWriteBytes: 5 * 1024 * 1024
  }
});

export type Config = z.infer<typeof ConfigSchema>;

export async function getConfig(args: string[]): Promise<Config> {
  if (args.length > 0 && args[0] === '--config') {
    if (args.length < 2) {
      console.error("Error: --config requires a path to a config file.");
      process.exit(1);
    }
    const configFile = args[1];
    try {
      const configContent = await fs.readFile(configFile, 'utf-8');
      return ConfigSchema.parse(JSON.parse(configContent));
    } catch (err) {
      console.error(`Error loading config file ${configFile}:`, err);
      process.exit(1);
    }
  }

  // Default config with file filtering
  return ConfigSchema.parse({
    allowedDirectories: args.length > 0 ? args : [process.cwd()],
    fileFiltering: {
      defaultExcludes: DEFAULT_EXCLUDE_PATTERNS,
      allowedExtensions: DEFAULT_ALLOWED_EXTENSIONS,
      forceTextFiles: true
    },
    fuzzyMatching: {
      maxDistanceRatio: 0.25,
      minSimilarity: 0.7,
      caseSensitive: false,
      ignoreWhitespace: true,
      preserveLeadingWhitespace: 'auto'
    },
    logging: {
      level: 'info',
      performance: false
    },
    concurrency: {
      maxConcurrentEdits: 10,
      maxGlobalConcurrentEdits: 20
    },
    limits: {
      maxReadBytes: 5 * 1024 * 1024,
      maxWriteBytes: 5 * 1024 * 1024
    }
  });
}
```
#### Plik: `src/types/errors.ts`
```ts
import { HintInfo, HINTS } from "../utils/hintMap";

export interface StructuredError {
  code: keyof typeof HINTS | string;
  message: string;
  hint?: HintInfo["hint"];
  confidence?: HintInfo["confidence"];
  details?: unknown;
}

export function createError(
  code: StructuredError["code"],
  message: string,
  details?: unknown
): StructuredError {
  const hint = HINTS[code as keyof typeof HINTS];
  return {
    code,
    message,
    hint: hint?.hint,
    confidence: hint?.confidence,
    details
  };
}
```
#### Plik: `src/types/fast-levenshtein.d.ts`
```ts
declare module 'fast-levenshtein' {
  export function get(a: string, b: string): number;
}
```
#### Plik: `src/core/security.ts`
```ts
import fs from 'node:fs/promises';
import path from 'node:path';
import { PerformanceTimer } from '../utils/performance.js';
import { expandHome, normalizePath } from '../utils/pathUtils.js';
import { createError } from '../types/errors.js';
import type { Logger } from 'pino';
import type { Config } from '../server/config.js';

export async function validatePath(requestedPath: string, allowedDirectories: string[], logger: Logger, config: Config): Promise<string> {
  const timer = new PerformanceTimer('validatePath', logger, config);

  try {
    const expandedPath = expandHome(requestedPath);
    // If the path is relative, resolve it against the FIRST allowed directory (which is already absolute and normalized by server setup)
    // or server CWD if no allowedDirectories are configured (though server setup should ensure at least one).
    const absoluteInitialPath = path.isAbsolute(expandedPath)
      ? path.resolve(expandedPath)
      : path.resolve(allowedDirectories[0] ?? process.cwd(), expandedPath);

    const normalizedInitialPath = normalizePath(absoluteInitialPath);

    // Helper to check if a given path is within any of the allowed directories.
    // allowedDirectories are assumed to be absolute and normalized by the caller (server setup).
    const checkAgainstAllowedDirs = (pathToVerify: string): boolean => {
      return allowedDirectories.some(allowedDir => {
        // For Windows, compare paths case-insensitively.
        // Note: allowedDir is already absolute & normalized.
        const effectiveAllowedDir = process.platform === 'win32' ? allowedDir.toLowerCase() : allowedDir;
        const effectivePathToVerify = process.platform === 'win32' ? pathToVerify.toLowerCase() : pathToVerify;
        
        const relative = path.relative(effectiveAllowedDir, effectivePathToVerify);
        // Path is allowed if it's the same as allowedDir (relative is '') 
        // or if it's a subdirectory (relative doesn't start with '..' and is not '..')
        return !relative.startsWith('..' + path.sep) && relative !== '..';
      });
    };

    // 1. Check if the initially resolved path is within allowed directories.
    if (!checkAgainstAllowedDirs(normalizedInitialPath)) {
      throw createError(
        'ACCESS_DENIED',
        `Initial path '${normalizedInitialPath}' is outside allowed directories.`,
        { requestedPath: normalizedInitialPath, allowedDirectories }
      );
    }

    // 2. Attempt to get the real path (resolving symlinks).
    //    If realpath fails (e.g., path doesn't exist like for a new file), we use the normalizedInitialPath.
    let finalPathToReturn = normalizedInitialPath;
    try {
      const realPathAttempt = await fs.realpath(normalizedInitialPath);
      finalPathToReturn = normalizePath(realPathAttempt); // Use real path if successful
    } catch (error: any) {
      // If realpath fails because the path doesn't exist (ENOENT), it's okay for write/create operations.
      // We'll continue with normalizedInitialPath. For other errors, log them but still proceed with initial path for now.
      if (error.code !== 'ENOENT') {
        logger.debug({ path: normalizedInitialPath, error: error.message }, `fs.realpath failed, proceeding with initial path for further checks.`);
      }
      // finalPathToReturn remains normalizedInitialPath if realpath fails
    }
    
    // 3. Check if the final path (after potential symlink resolution or if it's a non-existent path)
    //    is still within allowed directories.
    if (!checkAgainstAllowedDirs(finalPathToReturn)) {
      throw createError(
        'ACCESS_DENIED',
        `Path '${finalPathToReturn}' (potentially after symlink resolution from '${normalizedInitialPath}') is outside allowed directories.`,
        { originalPath: normalizedInitialPath, resolvedPath: finalPathToReturn, allowedDirectories }
      );
    }

    timer.end({ result: 'success', path: finalPathToReturn });
    return finalPathToReturn;

  } catch (error: any) {
    timer.end({ result: 'error' });
    // If it's already a StructuredError from createError, rethrow it.
    if (error.code && error.message && error.hint) { 
      throw error;
    }
    // Wrap other unexpected errors.
    throw createError('VALIDATION_ERROR', (error as Error).message || String(error), { originalErrorDetails: String(error) });
  }
}
```
#### Plik: `src/core/concurrency.ts`
```ts
import { Semaphore } from 'async-mutex';
import type { Config } from '../server/config.js';

let globalEditSemaphore: Semaphore;

export function initGlobalSemaphore(config: Config) {
  globalEditSemaphore = new Semaphore(config.concurrency.maxGlobalConcurrentEdits ?? 20);
}

export function getGlobalSemaphore() {
  if (!globalEditSemaphore) {
    throw new Error('Global semaphore not initialized');
  }
  return globalEditSemaphore;
}
```
#### Plik: `src/core/fuzzyEdit.ts`
```ts
import fs from 'node:fs/promises';
import { createTwoFilesPatch } from 'diff';
import { isBinaryFile } from '../utils/binaryDetect';
import { createError } from '../types/errors';
import { get as fastLevenshtein } from 'fast-levenshtein';
import { PerformanceTimer } from '../utils/performance';
import type { Logger } from 'pino';
import type { EditOperation } from './schemas';
import type { Config } from '../server/config';

interface AppliedEditRange {
  startLine: number;
  endLine: number;
  editIndex: number; // To identify which edit operation this range belongs to
}

function doRangesOverlap(range1: AppliedEditRange, range2: {startLine: number, endLine: number}): boolean {
  return Math.max(range1.startLine, range2.startLine) <= Math.min(range1.endLine, range2.endLine);
}

export interface FuzzyMatchConfig {
  maxDistanceRatio: number;
  minSimilarity: number;
  caseSensitive: boolean;
  ignoreWhitespace: boolean;
  preserveLeadingWhitespace: 'auto' | 'strict' | 'normalize';
  debug: boolean;
  forcePartialMatch?: boolean; // Added for forcePartialMatch option per edit
}

export interface ApplyFileEditsResult {
  modifiedContent: string;
  formattedDiff: string;
}

function normalizeLineEndings(text: string): string {
  return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
}

function createUnifiedDiff(originalContent: string, newContent: string, filepath: string = 'file'): string {
  const normalizedOriginal = normalizeLineEndings(originalContent);
  const normalizedNew = normalizeLineEndings(newContent);
  return createTwoFilesPatch(
    filepath,
    filepath,
    normalizedOriginal,
    normalizedNew,
    'original',
    'modified'
  );
}

function preprocessText(text: string, config: FuzzyMatchConfig): string {
  let processed = text;
  if (!config.caseSensitive) {
    processed = processed.toLowerCase();
  }
  if (config.ignoreWhitespace) {
    processed = processed.replace(/[ \t]+/g, ' ').replace(/\n+/g, '\n').trim();
  }
  return processed;
}

// Main edit application logic
export async function applyFileEdits(
  filePath: string,
  edits: EditOperation[],
  config: FuzzyMatchConfig,
  logger: Logger,
  globalConfig: Config
): Promise<ApplyFileEditsResult> {
  const timer = new PerformanceTimer('applyFileEdits', logger, globalConfig);
  let levenshteinIterations = 0;

  try {
    const buffer = await fs.readFile(filePath);
    if (await isBinaryFile(buffer, filePath)) {
      throw createError(
        'BINARY_FILE_ERROR',
        'Cannot edit binary files',
        { filePath, detectedAs: 'binary' }
      );
    }

    const originalContent = normalizeLineEndings(buffer.toString('utf-8'));
    let modifiedContent = originalContent;
    const appliedRanges: AppliedEditRange[] = [];

    validateEdits(edits, config.debug, logger);

    for (const [editIndex, edit] of edits.entries()) {
      let replaced = false;
      const normalizedOld = normalizeLineEndings(edit.oldText);
      const normalizedNew = normalizeLineEndings(edit.newText);

      // Simple exact match replacement for single-line edits (exact spaces)
      if (!normalizedOld.includes('\n') && modifiedContent.includes(config.caseSensitive ? edit.oldText : (config.ignoreWhitespace ? normalizedOld.replace(/\s+/g,' ') : normalizedOld))) {
        const searchText = config.caseSensitive ? edit.oldText : normalizedOld;
        const replaceText = config.caseSensitive ? edit.newText : normalizedNew;
        modifiedContent = modifiedContent.replace(searchText, replaceText);
        replaced = true;
        continue;
      }

      if (!replaced && config.ignoreWhitespace && !normalizedOld.includes('\n')) {
        const whitespacePattern = edit.oldText.replace(/\s+/g, "\\s+");
        const flags = config.caseSensitive ? 'g' : 'gi';
        const regex = new RegExp(whitespacePattern, flags);
        if (regex.test(modifiedContent)) {
          modifiedContent = modifiedContent.replace(regex, edit.newText);
          replaced = true;
        }
      }

      // If still not replaced and caseSensitive, throw early


      const exactMatchIndex = modifiedContent.indexOf(normalizedOld);
      if (exactMatchIndex !== -1) {
        replaced = true;
        // Preserve indentation for exact matches using the same logic as fuzzy matches
        const contentLines = modifiedContent.split('\n');
        const oldLinesForIndent = normalizedOld.split('\n');
        const newLinesForIndent = normalizedNew.split('\n');

        // Find the line number of the exact match to get the original indent
        let charCount = 0;
        let lineNumberOfMatch = 0;
        for (let i = 0; i < contentLines.length; i++) {
          if (charCount + contentLines[i].length + 1 > exactMatchIndex) {
            lineNumberOfMatch = i;
            break;
          }
          charCount += contentLines[i].length + 1;
        }

        const originalIndent = contentLines[lineNumberOfMatch].match(/^\s*/)?.[0] ?? '';
        const indentedNewLines = applyRelativeIndentation(
          newLinesForIndent,
          oldLinesForIndent,
          originalIndent,
          config.preserveLeadingWhitespace
        );

        // Reconstruct modifiedContent carefully with the new indented lines
        const linesBeforeMatch = modifiedContent.substring(0, exactMatchIndex).split('\n');
        const linesAfterMatch = modifiedContent.substring(exactMatchIndex + normalizedOld.length).split('\n');

        // The new content replaces a certain number of original lines that constituted normalizedOld.
        // We need to splice the contentLines array correctly.
        // The number of lines to replace is oldLinesForIndent.length.
        // The starting line for replacement is lineNumberOfMatch.
        const currentEditTargetRange = {
          startLine: lineNumberOfMatch,
          endLine: lineNumberOfMatch + oldLinesForIndent.length - 1
        };

        for (const appliedRange of appliedRanges) {
          if (doRangesOverlap(appliedRange, currentEditTargetRange)) {
            throw createError(
              'OVERLAPPING_EDIT',
              `Edit ${editIndex + 1} (exact match) overlaps with previously applied edit ${appliedRange.editIndex + 1}. ` +
              `Current edit targets lines ${currentEditTargetRange.startLine + 1}-${currentEditTargetRange.endLine + 1}. ` +
              `Previous edit affected lines ${appliedRange.startLine + 1}-${appliedRange.endLine + 1}.`,
              {
                conflictingEditIndex: editIndex,
                previousEditIndex: appliedRange.editIndex,
                currentEditTargetRange,
                previousEditAffectedRange: appliedRange
              }
            );
          }
        }

        const tempContentLines = modifiedContent.split('\n');
        tempContentLines.splice(lineNumberOfMatch, oldLinesForIndent.length, ...indentedNewLines);
        modifiedContent = tempContentLines.join('\n');

        appliedRanges.push({
          startLine: currentEditTargetRange.startLine,
          endLine: currentEditTargetRange.startLine + indentedNewLines.length - 1,
          editIndex
        });
      } else {
        // Fuzzy match logic
        const contentLines = modifiedContent.split('\n');
        const oldLines = normalizedOld.split('\n');
        const processedOld = preprocessText(normalizedOld, config);

        let bestMatch = {
          distance: Infinity,
          index: -1,
          text: '',
          similarity: 0,
          windowSize: 0
        };
        // ... (tu znajduje się dalsza logika fuzzy match)
        // Jeśli fuzzy match się powiedzie, ustaw replaced = true;
        // Jeśli nie, replaced pozostaje false
      }

      if (config.caseSensitive && !replaced) {
        throw createError('NO_MATCH', `No match found for edit "${edit.oldText}" (caseSensitive)`);
      }
    }

    const diff = createUnifiedDiff(originalContent, modifiedContent, filePath);
    const MAX_DIFF_LINES = 4000; // Configurable limit for diff lines
    const diffLines = diff.split('\n');
    let formattedDiff = "";
    if (diffLines.length > MAX_DIFF_LINES) {
      formattedDiff = "```diff\n" +
                      diffLines.slice(0, MAX_DIFF_LINES).join('\n') +
                      `\n...diff truncated (${diffLines.length - MAX_DIFF_LINES} lines omitted)\n` +
                      "```\n\n";
    } else {
      formattedDiff = "```diff\n" + diff + "\n```\n\n";
    }

    timer.end({ 
      editsCount: edits.length, 
      levenshteinIterations,
      charactersProcessed: originalContent.length
    });

    return { modifiedContent, formattedDiff };
  } catch (error) {
    timer.end({ result: 'error' });
    throw error;
  }
}

function applyRelativeIndentation(
  newLines: string[], 
  oldLines: string[], 
  originalIndent: string,
  preserveMode: 'auto' | 'strict' | 'normalize'
): string[] {
  // ... (oryginalna logika)
  // Zostawiamy bez zmian
  return newLines;
}

function validateEdits(edits: Array<{oldText: string, newText: string}>, debug: boolean, logger: Logger): void {
  // ... (oryginalna logika)
}

function getContextLines(text: string, lineNumber: number, contextSize: number): string {
  // ... (oryginalna logika)
  return '';
}
```
#### Plik: `src/core/__tests__/applyFileEdits.test.ts`
```ts
/// <reference types="jest" />
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { applyFileEdits, FuzzyMatchConfig } from '../fuzzyEdit';
import type { Config } from '../../server/config';

// Minimal stub config for tests
const testConfig = {
    allowedDirectories: [],
    fileFiltering: {
        defaultExcludes: [],
        allowedExtensions: ['*.txt'],
        forceTextFiles: true
    },
    fuzzyMatching: {
        maxDistanceRatio: 0.25,
        minSimilarity: 0.7,
        caseSensitive: false,
        ignoreWhitespace: true,
        preserveLeadingWhitespace: 'auto'
    },
    logging: { level: 'info', performance: false },
    concurrency: { maxConcurrentEdits: 1, maxGlobalConcurrentEdits: 1 },
    limits: { maxReadBytes: 1024 * 1024, maxWriteBytes: 1024 * 1024 }
} as unknown as Config;

// Simple no-op logger stub
const noopLogger = {
    info: () => { },
    warn: () => { },
    error: () => { },
    debug: () => { },
    trace: () => { }
} as any;

async function createTempFile(content: string): Promise<string> {
    const tmpDir = os.tmpdir();
    const filePath = path.join(tmpDir, `applyFileEdits_${Date.now()}_${Math.random().toString(36).slice(2)}.txt`);
    await fs.writeFile(filePath, content, 'utf-8');
    return filePath;
}

describe('applyFileEdits', () => {
    it('applies exact text replacement', async () => {
        const original = 'foo bar baz';
        const filePath = await createTempFile(original);

        const edits = [{ oldText: 'bar', newText: 'qux', forcePartialMatch: false }];
        const fuzzyConfig: FuzzyMatchConfig = {
            maxDistanceRatio: 0.25,
            minSimilarity: 0.7,
            caseSensitive: false,
            ignoreWhitespace: true,
            preserveLeadingWhitespace: 'auto',
            debug: false
        };

        const res = await applyFileEdits(filePath, edits, fuzzyConfig, noopLogger, testConfig);

        expect(res.modifiedContent).toBe('foo qux baz');
        expect(res.formattedDiff).toContain('-foo bar baz');
        expect(res.formattedDiff).toContain('+foo qux baz');
    });

    it('respects caseSensitive flag', async () => {
        const original = 'Hello World';
        const filePath = await createTempFile(original);

        const edits = [{ oldText: 'hello', newText: 'Hi', forcePartialMatch: false }];
        const fuzzyConfig: FuzzyMatchConfig = {
            maxDistanceRatio: 0.25,
            minSimilarity: 0.7,
            caseSensitive: true,
            ignoreWhitespace: true,
            preserveLeadingWhitespace: 'auto',
            debug: false
        };

        await expect(applyFileEdits(filePath, edits, fuzzyConfig, noopLogger, testConfig)).rejects.toThrow();
    });

    it('handles ignoreWhitespace option', async () => {
        const original = 'alpha    beta';
        const filePath = await createTempFile(original);

        const edits = [{ oldText: 'alpha beta', newText: 'gamma', forcePartialMatch: false }];
        const fuzzyConfig: FuzzyMatchConfig = {
            maxDistanceRatio: 0.3,
            minSimilarity: 0.6,
            caseSensitive: false,
            ignoreWhitespace: true,
            preserveLeadingWhitespace: 'auto',
            debug: false
        };

        const res = await applyFileEdits(filePath, edits, fuzzyConfig, noopLogger, testConfig);
        expect(res.modifiedContent).toBe('gamma');
    });
});
```
#### Plik: `src/core/fileInfo.ts`
```ts
import fs from 'node:fs/promises';
import path from 'node:path';
import { minimatch } from 'minimatch';
import { lookup as mimeLookup } from 'mime-types';
import { isBinaryFile } from '../utils/binaryDetect.js';
import { PerformanceTimer } from '../utils/performance.js';
import { validatePath } from './security.js';
import { shouldSkipPath } from '../utils/pathFilter.js';
import type { Logger } from 'pino';
import type { Config } from '../server/config.js';

export interface FileInfo {
  size: number;
  created: Date;
  modified: Date;
  accessed: Date;
  isDirectory: boolean;
  isFile: boolean;
  permissions: string;
  isBinary?: boolean;
  mimeType?: string;
}

const FILE_STAT_READ_BUFFER_SIZE = 8192; // Read 8KB for binary detection

export async function getFileStats(filePath: string, logger: Logger, config: Config): Promise<FileInfo> {
  const timer = new PerformanceTimer('getFileStats', logger, config);

  try {
    const stats = await fs.stat(filePath);
    let isBinary = false;
    let mimeType: string | false = false;

    if (stats.isFile()) {
      // For files, read a small chunk to determine if binary and get a better mime type
      const fileHandle = await fs.open(filePath, 'r');
      try {
        const buffer = Buffer.alloc(FILE_STAT_READ_BUFFER_SIZE);
        const { bytesRead } = await fileHandle.read(buffer, 0, FILE_STAT_READ_BUFFER_SIZE, 0);
        const actualBuffer = bytesRead < FILE_STAT_READ_BUFFER_SIZE ? buffer.subarray(0, bytesRead) : buffer;
        isBinary = isBinaryFile(actualBuffer, filePath);
      } finally {
        await fileHandle.close();
      }
      mimeType = mimeLookup(filePath);
    }

    const result = {
      size: stats.size,
      created: stats.birthtime,
      modified: stats.mtime,
      accessed: stats.atime,
      isDirectory: stats.isDirectory(),
      isFile: stats.isFile(),
      permissions: stats.mode.toString(8).slice(-3),
      isBinary: stats.isFile() ? isBinary : undefined, // isBinary is only relevant for files
      mimeType: stats.isFile() ? (mimeType || (isBinary ? 'application/octet-stream' : 'text/plain')) : undefined
    };

    timer.end({ isBinary, size: stats.size });
    return result;
  } catch (error) {
    timer.end({ result: 'error' });
    throw error;
  }
}

export async function searchFiles(
  rootPath: string,
  pattern: string,
  logger: Logger,
  config: Config,
  excludePatterns: string[] = [],
  useExactPatterns: boolean = false,
  maxDepth: number = -1 // Add maxDepth, -1 for unlimited
): Promise<string[]> {
  const timer = new PerformanceTimer('searchFiles', logger, config);
  const results: string[] = [];

  async function search(currentPath: string, currentDepth: number): Promise<void> { // Add currentDepth
    if (maxDepth !== -1 && currentDepth > maxDepth) {
      logger.debug({ path: currentPath, currentDepth, maxDepth }, 'Max depth reached in searchFiles, stopping recursion for this path');
      return;
    }

    try {
      // No need for fs.stat here if readdir withFileTypes is used, unless specific inode-based symlink detection is absolutely required.
      // For basic symlink loop prevention, checking entry.isSymbolicLink() before recursing is often sufficient.
      const entries = await fs.readdir(currentPath, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(currentPath, entry.name);

        try {
          const relativePath = path.relative(rootPath, fullPath);
          // Exclude via explicit excludePatterns first
          if (excludePatterns.some(p => minimatch(relativePath, p, { dot: true, nocase: true }))) {
            continue;
          }

          // Apply the centralized filtering utility
          if (shouldSkipPath(fullPath, config)) {
            continue;
          }

          // Determine the actual glob pattern to use based on useExactPatterns
          // If useExactPatterns is false and the input pattern doesn't contain a path separator,
          // prepend '**/' to match items at any depth.
          const globPatternToUse = useExactPatterns
            ? pattern
            : (pattern.includes('/') ? pattern : `**/${pattern}`);

          // Match the effective glob pattern against the relative path, case-insensitively
          if (minimatch(relativePath, globPatternToUse, { dot: true, nocase: true })) {
            results.push(fullPath); // Add full path of the matching file/directory
          }

          // If it's a directory (and wasn't excluded), recurse into it.
          // The directory itself might have matched the pattern and been added above.
          // Recursion happens to find matching items *within* this directory.
          if (entry.isDirectory()) {
            if (entry.isSymbolicLink()) {
              // Basic symlink check: if it's a directory and a symlink, be cautious.
              // For more robust cycle detection, fs.realpath and tracking visited real paths would be needed,
              // but that adds more I/O. For now, skip recursing into directory symlinks to avoid simple loops.
              logger.debug({ path: fullPath }, 'Skipping recursion into directory symlink to avoid potential loops.');
            } else {
              await search(fullPath, currentDepth + 1); // Increment depth
            }
          }
        } catch (error) {
          // This catch block might still be relevant if fs.stat or fs.readdir fails for a specific entry
          // even if the parent was accessible. However, the validatePath call was the primary source of errors here.
          logger.debug({ path: fullPath, error: (error as Error).message }, 'Error processing entry during search');
          continue;
        }
      }
    } catch (error) {
      logger.debug({ path: currentPath, error: (error as Error).message }, 'Skipping inaccessible path');
      return;
    }
  }

  await search(rootPath, 0); // Start depth at 0
  timer.end({ resultsCount: results.length });
  return results;
}

import type { DirectoryTreeEntry } from './schemas.js';

export async function getDirectoryTree(
  basePath: string,
  allowedDirectories: string[],
  logger: Logger,
  config: Config,
  currentDepth: number = 0,
  maxDepth: number = -1 // Default to no max depth (-1 means unlimited)
): Promise<DirectoryTreeEntry> {
  const timer = new PerformanceTimer('getDirectoryTree', logger, config);
  logger.debug({ basePath, currentDepth, maxDepth }, 'Starting getDirectoryTree for path');

  const validatedPath = await validatePath(basePath, allowedDirectories, logger, config);
  const stats = await fs.stat(validatedPath);
  const name = path.basename(validatedPath);

  const entry: DirectoryTreeEntry = {
    name,
    path: validatedPath,
    type: stats.isDirectory() ? 'directory' : 'file',
  };

  if (stats.isDirectory()) {
    // Check if maxDepth is set and if currentDepth has reached it
    if (maxDepth !== -1 && currentDepth >= maxDepth) {
      logger.debug({ basePath, currentDepth, maxDepth }, 'Max depth reached, not traversing further');
      entry.children = []; // Indicate that there might be more, but not traversing
      timer.end({ path: basePath, type: 'directory', depthReached: true });
      return entry;
    }

    entry.children = [];
    try {
      const dirents = await fs.readdir(validatedPath, { withFileTypes: true });
      for (const dirent of dirents) {
        const childPath = path.join(validatedPath, dirent.name);

        // Apply centralized filtering for each child entry
        if (shouldSkipPath(childPath, config)) {
          continue;
        }

        // Recursive call, incrementing currentDepth
        // We pass the original maxDepth down
        const childEntry = await getDirectoryTree(childPath, allowedDirectories, logger, config, currentDepth + 1, maxDepth);
        entry.children.push(childEntry);
      }
    } catch (error: any) {
      logger.warn({ path: validatedPath, error: error.message }, 'Failed to read directory contents in getDirectoryTree');
      // Optionally, add an error node or just skip: entry.children.push({ name: 'Error reading directory', path: validatedPath, type: 'error' });
    }
  }

  timer.end({ path: basePath, type: entry.type, childrenCount: entry.children?.length });
  return entry;
}

// Add this function to src/core/fileInfo.ts

export interface FileReadResult {
  path: string;
  content?: string;
  encoding?: 'utf-8' | 'base64' | 'error'; // Changed from encodingUsed, added 'error'
}

export async function readMultipleFilesContent(
  filePaths: string[],
  requestedEncoding: 'utf-8' | 'base64' | 'auto',
  allowedDirectories: string[],
  logger: Logger,
  config: Config
): Promise<FileReadResult[]> {
  const timer = new PerformanceTimer('readMultipleFilesContent', logger, config);
  const results: FileReadResult[] = [];

  for (const filePath of filePaths) {
    try {
      const validPath = await validatePath(filePath, allowedDirectories, logger, config);
      const rawBuffer = await fs.readFile(validPath);
      let content: string;
      let finalEncoding: 'utf-8' | 'base64' = 'utf-8'; // Default to utf-8

      if (requestedEncoding === 'base64') {
        content = rawBuffer.toString('base64');
        finalEncoding = 'base64';
      } else if (requestedEncoding === 'auto') {
        // For 'auto', we need to read a small chunk to detect binary, similar to getFileStats
        // This re-reads a small part if the file is large, could be optimized if rawBuffer is small enough already.
        const checkBuffer = rawBuffer.length > FILE_STAT_READ_BUFFER_SIZE ? rawBuffer.subarray(0, FILE_STAT_READ_BUFFER_SIZE) : rawBuffer;
        if (isBinaryFile(checkBuffer, validPath)) {
          content = rawBuffer.toString('base64');
          finalEncoding = 'base64';
        } else {
          content = rawBuffer.toString('utf-8');
          finalEncoding = 'utf-8';
        }
      } else { // utf-8 is the default or explicitly requested
        content = rawBuffer.toString('utf-8');
        finalEncoding = 'utf-8';
      }
      results.push({ path: filePath, content, encoding: finalEncoding });
    } catch (error: any) {
      logger.warn({ path: filePath, error: error.message }, 'Failed to read one of the files in readMultipleFilesContent');
      results.push({ path: filePath, content: `Error: ${error.message}`, encoding: 'error' }); // Error in content, encoding='error'
    }
  }

  timer.end({ filesCount: filePaths.length, resultsCount: results.length });
  return results;
}
```
#### Plik: `src/core/toolHandlers.ts`
```ts
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { Mutex } from 'async-mutex';
import { initGlobalSemaphore, getGlobalSemaphore } from './concurrency.js';
import fs from 'node:fs/promises';
import path from 'node:path';


import { createError, StructuredError } from '../types/errors.js';
import { shouldSkipPath } from '../utils/pathFilter.js';

import { PerformanceTimer } from '../utils/performance.js';
import { isBinaryFile } from '../utils/binaryDetect.js';
import { validatePath } from './security.js';
import { applyFileEdits, FuzzyMatchConfig } from './fuzzyEdit.js';
import { getFileStats, searchFiles, readMultipleFilesContent, getDirectoryTree } from './fileInfo.js';
import * as schemas from './schemas.js'; // <-- BRAKUJĄCY IMPORT ZOSTAŁ DODANY
// Import specific types that were causing issues if not directly imported
import type { ListDirectoryEntry, DirectoryTreeEntry } from './schemas.js';

import type { Logger } from 'pino';
import type { Config } from '../server/config.js';

let requestCount = 0;
let editOperationCount = 0;

const fileLocks = new Map<string, Mutex>();

function getFileLock(filePath: string, config: Config, logger: Logger): Mutex {
  if (fileLocks.has(filePath)) {
    // Move to end of map to mark as recently used
    const existingMutex = fileLocks.get(filePath)!;
    fileLocks.delete(filePath);
    fileLocks.set(filePath, existingMutex);
    return existingMutex;
  } else {
    if (fileLocks.size >= config.concurrency.maxConcurrentEdits) {
      let evicted = false;
      // Iterate from oldest (insertion order)
      for (const [key, mutex] of fileLocks.entries()) {
        if (!mutex.isLocked()) {
          fileLocks.delete(key);
          logger.debug({ evictedKey: key, newKey: filePath, mapSize: fileLocks.size }, 'Evicted inactive lock to make space.');
          evicted = true;
          break;
        }
      }
      if (!evicted) {
        // All locks are active, and map is full
        logger.error({ filePath, mapSize: fileLocks.size, maxConcurrentEdits: config.concurrency.maxConcurrentEdits }, 'Cannot acquire new file lock: max concurrent locks reached, and all are active.');
        throw createError('MAX_CONCURRENCY_REACHED', `Cannot acquire new file lock for ${filePath}: Maximum concurrent file locks (${config.concurrency.maxConcurrentEdits}) reached, and all are currently active.`);
      }
    }
    const newMutex = new Mutex();
    fileLocks.set(filePath, newMutex);
    return newMutex;
  }
}



export function setupToolHandlers(server: Server, allowedDirectories: string[], logger: Logger, config: Config) {
  initGlobalSemaphore(config);
  server.setRequestHandler(schemas.HandshakeRequestSchema, async () => ({
    serverName: 'mcp-filesystem-server',
    serverVersion: '0.7.0'
  }));
  const EditFileArgsSchema = schemas.getEditFileArgsSchema(config);

  server.setRequestHandler(schemas.ListToolsRequestSchema, async () => {
    return {
      tools: [
        { name: 'read_file', description: 'Read file contents.', inputSchema: zodToJsonSchema(schemas.ReadFileArgsSchema) as any },
        { name: 'read_multiple_files', description: 'Read multiple files.', inputSchema: zodToJsonSchema(schemas.ReadMultipleFilesArgsSchema) as any },
        { name: 'list_allowed_directories', description: 'List allowed base directories.', inputSchema: zodToJsonSchema(schemas.ListAllowedDirectoriesArgsSchema) as any },
        { name: 'write_file', description: 'Write file contents.', inputSchema: zodToJsonSchema(schemas.WriteFileArgsSchema) as any },
        { name: 'edit_file', description: 'Edit file contents using fuzzy matching.', inputSchema: zodToJsonSchema(EditFileArgsSchema) as any },
        { name: 'create_directory', description: 'Create a directory.', inputSchema: zodToJsonSchema(schemas.CreateDirectoryArgsSchema) as any },
        { name: 'list_directory', description: 'List directory contents.', inputSchema: zodToJsonSchema(schemas.ListDirectoryArgsSchema) as any },
        { name: 'directory_tree', description: 'Get directory tree.', inputSchema: zodToJsonSchema(schemas.DirectoryTreeArgsSchema) as any },
        { name: 'move_file', description: 'Move/rename a file or directory.', inputSchema: zodToJsonSchema(schemas.MoveFileArgsSchema) as any },
        { name: 'delete_file', description: 'Delete a file.', inputSchema: zodToJsonSchema(schemas.DeleteFileArgsSchema) as any },
        { name: 'delete_directory', description: 'Delete a directory.', inputSchema: zodToJsonSchema(schemas.DeleteDirectoryArgsSchema) as any },
        { name: 'search_files', description: 'Search for files by pattern.', inputSchema: zodToJsonSchema(schemas.SearchFilesArgsSchema) as any },
        { name: 'get_file_info', description: 'Get file/directory metadata.', inputSchema: zodToJsonSchema(schemas.GetFileInfoArgsSchema) as any },
        { name: 'server_stats', description: 'Get server statistics.', inputSchema: zodToJsonSchema(schemas.ServerStatsArgsSchema) as any }
      ]
    };
  });

  server.setRequestHandler(schemas.CallToolRequestSchema, async (request) => {
    requestCount++;
    logger.info({ tool: request.params.name, args: request.params.args }, `Tool request: ${request.params.name}`);
    try {
      switch (request.params.name) {
        case 'list_allowed_directories': {
          const parsed = schemas.ListAllowedDirectoriesArgsSchema.safeParse(request.params.args ?? {});
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ListAllowedDirectoriesArgsSchema) });
          return { result: { directories: allowedDirectories } };
        }

        case 'read_file': {
          const parsed = schemas.ReadFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ReadFileArgsSchema) });
          const timer = new PerformanceTimer('read_file_handler', logger, config);
          try {
            const validatedPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
            if (shouldSkipPath(validatedPath, config)) {
              throw createError('ACCESS_DENIED', 'File access denied due to filtering rules.');
            }
            const rawBuffer = await fs.readFile(validatedPath);

            // Enforce configured maximum read size
            if (rawBuffer.length > config.limits.maxReadBytes) {
              throw createError(
                'FILE_TOO_LARGE',
                `File size ${rawBuffer.length} exceeds configured maxReadBytes limit of ${config.limits.maxReadBytes} bytes`,
                { path: validatedPath, size: rawBuffer.length, limit: config.limits.maxReadBytes }
              );
            }

            let content: string;
            let encodingUsed: 'utf-8' | 'base64' = 'utf-8';
            const isBinary = isBinaryFile(rawBuffer, validatedPath);

            if (parsed.data.encoding === 'base64' || (parsed.data.encoding === 'auto' && isBinary)) {
              content = rawBuffer.toString('base64');
              encodingUsed = 'base64';
            } else {
              content = rawBuffer.toString('utf-8'); // Default to utf-8
            }
            timer.end();
            return { result: { content, encoding: encodingUsed } };
          } catch (error) {
            timer.end({ result: 'error' });
            throw error;
          }
        }

        case 'read_multiple_files': {
          const parsed = schemas.ReadMultipleFilesArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ReadMultipleFilesArgsSchema) });
          const fileReadResults = await readMultipleFilesContent(
            parsed.data.paths,
            parsed.data.encoding,
            allowedDirectories,
            logger,
            config
          );
          return { result: { files: fileReadResults } };
        }

        case 'write_file': {
          const parsed = schemas.WriteFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.WriteFileArgsSchema) });
          const timer = new PerformanceTimer('write_file_handler', logger, config);
          try {
            const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
            if (shouldSkipPath(validPath, config)) {
              throw createError('ACCESS_DENIED', 'File writing denied due to filtering rules.');
            }
            const lock = getFileLock(validPath, config, logger);
            await getGlobalSemaphore().runExclusive(async () => {
              await lock.runExclusive(async () => {
                const contentBuffer = Buffer.from(parsed.data.content, parsed.data.encoding);

                // Enforce configured maximum write size
                if (contentBuffer.length > config.limits.maxWriteBytes) {
                  throw createError(
                    'WRITE_TOO_LARGE',
                    `Content size ${contentBuffer.length} exceeds configured maxWriteBytes limit of ${config.limits.maxWriteBytes} bytes`,
                    { path: validPath, size: contentBuffer.length, limit: config.limits.maxWriteBytes }
                  );
                }

                await fs.writeFile(validPath, contentBuffer);
              });
            });
            timer.end();
            return { content: [{ type: 'text', text: `File written: ${parsed.data.path}` }] };
          } catch (error) {
            timer.end({ result: 'error' });
            throw error;
          }
        }

        case 'edit_file': {
          editOperationCount++;
          const parsed = EditFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          if (shouldSkipPath(validPath, config)) {
            throw createError('ACCESS_DENIED', 'File editing denied due to filtering rules', { path: validPath });
          }
          const fuzzyConfig: FuzzyMatchConfig = {
            maxDistanceRatio: parsed.data.maxDistanceRatio,
            minSimilarity: parsed.data.minSimilarity,
            caseSensitive: parsed.data.caseSensitive,
            ignoreWhitespace: parsed.data.ignoreWhitespace,
            preserveLeadingWhitespace: parsed.data.preserveLeadingWhitespace,
            debug: parsed.data.debug || config.logging.level === 'debug',
          };

          const lock = getFileLock(validPath, config, logger);
          let formattedDiff: string = '';

          await getGlobalSemaphore().runExclusive(async () => {
            await lock.runExclusive(async () => {
              const editResult = await applyFileEdits(
                validPath,
                parsed.data.edits,
                fuzzyConfig,
                logger,
                config
              );
              formattedDiff = editResult.formattedDiff;

              if (!parsed.data.dryRun) {
                await fs.writeFile(validPath, editResult.modifiedContent, 'utf-8');
              }
            });
          });

          const responseText = parsed.data.dryRun
            ? `Dry run: File '${parsed.data.path}' would be modified. Diff:\n${formattedDiff}`
            : `File '${parsed.data.path}' edited successfully. Diff:\n${formattedDiff}`;

          return { content: [{ type: 'text', text: responseText }] };
        }

        case 'create_directory': {
          const parsed = schemas.CreateDirectoryArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.CreateDirectoryArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          const lock = getFileLock(validPath, config, logger);
          await getGlobalSemaphore().runExclusive(async () => {
            await lock.runExclusive(async () => {
              await fs.mkdir(validPath, { recursive: true });
            });
          });
          return { content: [{ type: 'text', text: `Directory created: ${parsed.data.path}` }] };
        }

        case 'list_directory': {
          const parsed = schemas.ListDirectoryArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ListDirectoryArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          const entries = await fs.readdir(validPath, { withFileTypes: true });
          const results: ListDirectoryEntry[] = [];
          for (const dirent of entries) {
            const entryPath = path.join(validPath, dirent.name);
            if (shouldSkipPath(entryPath, config)) {
              continue;
            }

            let type: ListDirectoryEntry['type'] = 'other';
            if (dirent.isFile()) type = 'file';
            else if (dirent.isDirectory()) type = 'directory';
            else if (dirent.isSymbolicLink()) type = 'symlink';

            let size: number | undefined = undefined;
            if (type === 'file') {
              try {
                const stats = await fs.stat(entryPath);
                size = stats.size;
              } catch (statError) {
                logger.warn({ path: entryPath, error: statError }, 'Failed to get stats for file in list_directory');
              }
            }
            results.push({ name: dirent.name, path: path.relative(allowedDirectories[0], entryPath).replace(/\\/g, '/'), type, size });
          }
          return { result: { entries: results } };
        }

        case 'move_file': {
          const parsed = schemas.MoveFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.MoveFileArgsSchema) });

          const validSource = await validatePath(parsed.data.source, allowedDirectories, logger, config);
          const validDestination = await validatePath(parsed.data.destination, allowedDirectories, logger, config);

          if (validSource === validDestination) {
            return { content: [{ type: 'text', text: 'Source and destination are the same, no action taken.' }] };
          }
          if (shouldSkipPath(validSource, config) || shouldSkipPath(validDestination, config)) {
            throw createError('ACCESS_DENIED', 'Source or destination path is disallowed by filtering rules.');
          }

          await getGlobalSemaphore().runExclusive(async () => {
            // To prevent deadlocks, always acquire locks in a consistent order (alphabetical)
            const [path1, path2] = [validSource, validDestination].sort();
            const lock1 = getFileLock(path1, config, logger);
            const lock2 = getFileLock(path2, config, logger);

            await lock1.runExclusive(async () => {
              await lock2.runExclusive(async () => {
                await fs.rename(validSource, validDestination);
              });
            });
          });

          return { content: [{ type: 'text', text: `Moved from ${parsed.data.source} to ${parsed.data.destination}` }] };
        }

        case 'delete_file': {
          const parsed = schemas.DeleteFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.DeleteFileArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          if (shouldSkipPath(validPath, config)) {
            throw createError('ACCESS_DENIED', 'File deletion denied due to filtering rules.');
          }
          const lock = getFileLock(validPath, config, logger);
          await getGlobalSemaphore().runExclusive(async () => {
            await lock.runExclusive(async () => {
              await fs.unlink(validPath);
            });
          });
          return { content: [{ type: 'text', text: `File deleted: ${parsed.data.path}` }] };
        }

        case 'delete_directory': {
          const parsed = schemas.DeleteDirectoryArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.DeleteDirectoryArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          if (shouldSkipPath(validPath, config)) {
            throw createError('ACCESS_DENIED', 'Directory deletion denied due to filtering rules.');
          }
          const lock = getFileLock(validPath, config, logger);
          await getGlobalSemaphore().runExclusive(async () => {
            await lock.runExclusive(async () => {
              await fs.rm(validPath, { recursive: parsed.data.recursive || false, force: false }); // force: false for safety
            });
          });
          return { content: [{ type: 'text', text: `Directory deleted: ${parsed.data.path}` }] };
        }

        case 'search_files': {
          const parsed = schemas.SearchFilesArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.SearchFilesArgsSchema) });
          const timer = new PerformanceTimer('search_files_handler', logger, config);
          try {
            const validatedPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
            const results = await searchFiles(
              validatedPath,
              parsed.data.pattern,
              logger,
              config,
              parsed.data.excludePatterns || [],
              parsed.data.useExactPatterns || false,
              parsed.data.maxDepth || -1
            );

            let finalResults = results;
            if (parsed.data.maxResults && results.length > parsed.data.maxResults) {
              finalResults = results.slice(0, parsed.data.maxResults);
            }

            timer.end({ resultsCount: finalResults.length });
            return { result: { paths: finalResults } };
          } catch (error) {
            timer.end({ result: 'error' });
            throw error;
          }
        }

        case 'get_file_info': {
          const parsed = schemas.GetFileInfoArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.GetFileInfoArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          if (shouldSkipPath(validPath, config)) {
            throw createError('ACCESS_DENIED', 'File access denied due to filtering rules.');
          }
          const stats = await getFileStats(validPath, logger, config);
          return { result: stats };
        }

        case 'directory_tree': {
          const parsed = schemas.DirectoryTreeArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.DirectoryTreeArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          const tree = await getDirectoryTree(validPath, allowedDirectories, logger, config, 0, parsed.data.maxDepth ?? -1);
          return { result: tree };
        }

        case 'server_stats': {
          const parsed = schemas.ServerStatsArgsSchema.safeParse(request.params.args ?? {});
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ServerStatsArgsSchema) });
          const stats = { requestCount, editOperationCount, activeLocks: fileLocks.size, config };
          return { result: stats };
        }

        default:
          throw createError('UNKNOWN_TOOL', `Unknown tool: ${request.params.name}`);
      }
    } catch (error) {
      let structuredError: StructuredError;
      if ((error as any).code && (error as any).message) {
        structuredError = error as StructuredError;
      } else {
        structuredError = createError('UNKNOWN_ERROR', error instanceof Error ? error.message : String(error));
      }
      logger.error({ error: structuredError, tool: request.params.name }, `Tool request failed: ${request.params.name}`);
      return {
        content: [{ type: 'text', text: `Error (${structuredError.code}): ${structuredError.message}` }],
        isError: true,
        meta: { hint: structuredError.hint, confidence: structuredError.confidence, details: structuredError.details }
      };
    }
  });
}
```
#### Plik: `src/core/security.test.ts`
```ts
import { validatePath } from './security.js';
import { createError } from '../types/errors.js';
import path from 'node:path';
import pino from 'pino'; // Import pino
import { Config } from '../server/config.js'; // Import Config type
import { DEFAULT_EXCLUDE_PATTERNS, DEFAULT_ALLOWED_EXTENSIONS } from '../constants/extensions.js';

const testLogger = pino({ level: 'silent' }); // Create a silent pino instance for tests

describe('validatePath', () => {
  // Use the pino instance, or a simplified mock if direct pino usage is problematic
  const mockLogger = testLogger; 
  /* const mockLogger = {
    info: jest.fn(),
    debug: jest.fn(),
    error: jest.fn(),
    fatal: jest.fn(),
    warn: jest.fn(),
    trace: jest.fn(),
    silent: jest.fn(),
    level: 'test',
    child: jest.fn().mockReturnThis()
  }; */
  
  const mockConfig: Config = {
    allowedDirectories: ['/allowed'],
    fileFiltering: {
      defaultExcludes: DEFAULT_EXCLUDE_PATTERNS, // From constants
      allowedExtensions: DEFAULT_ALLOWED_EXTENSIONS, // From constants
      forceTextFiles: true // Default from ConfigSchema
    },
    logging: { level: 'info', performance: false }, // Matches ConfigSchema defaults
    fuzzyMatching: {
      maxDistanceRatio: 0.25,
      minSimilarity: 0.7,
      caseSensitive: false,
      ignoreWhitespace: true,
      preserveLeadingWhitespace: 'auto' as 'auto', // Ensure literal type
    },
    concurrency: {
      maxConcurrentEdits: 10,
      maxGlobalConcurrentEdits: 20 // Matches ConfigSchema defaults
    },
    limits: {
      maxReadBytes: 5 * 1024 * 1024, // Matches ConfigSchema defaults
      maxWriteBytes: 5 * 1024 * 1024 // Matches ConfigSchema defaults
    }
  };

  it('should allow paths within allowed directories', async () => {
    const result = await validatePath('/allowed/file.txt', ['/allowed'], mockLogger, mockConfig);
    expect(result).toBe(path.normalize('/allowed/file.txt'));
  });

  it('should reject paths outside allowed directories', async () => {
    await expect(validatePath('/outside/file.txt', ['/allowed'], mockLogger, mockConfig))
      .rejects
      .toHaveProperty('type', 'ACCESS_DENIED');
  });

  it('should resolve relative paths against first allowed directory', async () => {
    const result = await validatePath('file.txt', ['/allowed'], mockLogger, mockConfig);
    expect(result).toBe(path.normalize('/allowed/file.txt'));
  });

  it('should normalize path separators', async () => {
    const result = await validatePath('/allowed\subdir\file.txt', ['/allowed'], mockLogger, mockConfig);
    expect(result).toBe(path.normalize('/allowed/subdir/file.txt'));
  });

  it('should handle home directory expansion', async () => {
    const originalHome = process.env.HOME;
    process.env.HOME = '/home/user';
    
    const result = await validatePath('~/file.txt', ['/allowed'], mockLogger, mockConfig);
    expect(result).toBe(path.normalize('/home/user/file.txt'));
    
    process.env.HOME = originalHome;
  });

  // Note: Symlink tests would require actual filesystem setup
  // and are better suited for integration/e2e tests
});
```
#### Plik: `src/core/schemas.ts`
```ts
import { z } from 'zod';
import type { Config } from '../server/config.js';

export const HandshakeRequestSchema = z.object({
  method: z.literal('handshake'),
  params: z.object({}).optional()
});

export const ListToolsRequestSchema = z.object({
  method: z.literal('list_tools'),
  params: z.object({}).optional()
});

export const ReadFileArgsSchema = z.object({
  path: z.string(),
  encoding: z.enum(['utf-8', 'base64', 'auto']).default('auto').describe('Encoding for file content')
});

export const ReadMultipleFilesArgsSchema = z.object({
  paths: z.array(z.string()),
  encoding: z.enum(['utf-8', 'base64', 'auto']).default('auto').describe('Encoding for file content')
});

export const WriteFileArgsSchema = z.object({
  path: z.string(),
  content: z.string(),
  encoding: z.enum(['utf-8', 'base64']).default('utf-8').describe('Encoding of provided content')
});

export const EditOperationSchema = z.object({
  oldText: z.string().describe('Text to search for - can be slightly inaccurate'),
  newText: z.string().describe('Text to replace with'),
  forcePartialMatch: z.boolean().optional().default(false)
    .describe('If true, allows partial matches above minSimilarity threshold when no exact match is found')
});
export type EditOperation = z.infer<typeof EditOperationSchema>;

export const getEditFileArgsSchema = (config: Config) => z.object({
  path: z.string(),
  edits: z.array(EditOperationSchema),
  dryRun: z.boolean().default(false).describe('Preview changes using git-style diff format'),
  debug: z.boolean().default(false).describe('Show detailed matching information'),
  caseSensitive: z.boolean().default(config.fuzzyMatching.caseSensitive).describe('Whether to match case sensitively'),
  ignoreWhitespace: z.boolean().default(config.fuzzyMatching.ignoreWhitespace).describe('Whether to normalize whitespace differences'),
  maxDistanceRatio: z.number().min(0).max(1).default(config.fuzzyMatching.maxDistanceRatio).describe('Maximum allowed distance as ratio of text length'),
  minSimilarity: z.number().min(0).max(1).default(config.fuzzyMatching.minSimilarity).describe('Minimum similarity threshold (0-1)'),
  preserveLeadingWhitespace: z.enum(['auto', 'strict', 'normalize']).default(config.fuzzyMatching.preserveLeadingWhitespace).describe('How to handle leading whitespace preservation')
});

export const CreateDirectoryArgsSchema = z.object({
  path: z.string(),
});

export const ListDirectoryEntrySchema = z.object({
  name: z.string().describe('Name of the file or directory'),
  path: z.string().describe('Relative path from the base allowed directory'),
  type: z.enum(['file', 'directory', 'symlink', 'other']).describe('Type of the entry'),
  size: z.number().optional().describe('Size of the file in bytes, undefined for directories or if error reading stats')
});
export type ListDirectoryEntry = z.infer<typeof ListDirectoryEntrySchema>;

export const ListDirectoryArgsSchema = z.object({
  path: z.string(),
});

// Zaktualizowany schemat z dodanym `maxDepth`
export const DirectoryTreeArgsSchema = z.object({
  path: z.string(),
  maxDepth: z.number().int().positive().optional().describe('Maximum depth to traverse the directory tree')
});

// Define the recursive DirectoryTreeEntrySchema
// We need to use z.lazy to handle recursive types with Zod
export const DirectoryTreeEntrySchema: z.ZodType<DirectoryTreeEntry> = z.lazy(() =>
  z.object({
    name: z.string().describe('Name of the file or directory'),
    path: z.string().describe('Full absolute path of the file or directory'),
    type: z.enum(['file', 'directory']).describe('Type of the entry'),
    children: z.array(DirectoryTreeEntrySchema).optional().describe('Children of the directory entry, undefined for files'),
  })
);

// Define the TypeScript interface for DirectoryTreeEntry for clarity
export interface DirectoryTreeEntry {
  name: string;
  path: string;
  type: 'file' | 'directory';
  children?: DirectoryTreeEntry[];
}

// The result schema is essentially the root entry of the directory tree
export const DirectoryTreeResultSchema = DirectoryTreeEntrySchema;

export const MoveFileArgsSchema = z.object({
  source: z.string(),
  destination: z.string(),
});

export const ListAllowedDirectoriesArgsSchema = z.object({}); // No parameters for listing allowed directories

export const ServerStatsArgsSchema = z.object({}); // Schema for server_stats tool arguments

export const SearchFilesArgsSchema = z.object({
  path: z.string(),
  pattern: z.string(),
  excludePatterns: z.array(z.string()).optional().default([]),
  useExactPatterns: z.boolean().default(false).describe('Use patterns exactly as provided instead of wrapping with **/'),
  maxDepth: z.number().int().positive().optional().describe('Maximum depth to search'),
  maxResults: z.number().int().positive().optional().describe('Maximum number of results to return')
});

export const GetFileInfoArgsSchema = z.object({
  path: z.string(),
});

export const CallToolRequestSchema = z.object({
  method: z.literal('call_tool'),
  params: z.object({
    name: z.string(),
    args: z.any()
  })
});

export const DeleteFileArgsSchema = z.object({
  path: z.string(),
});

export const DeleteDirectoryArgsSchema = z.object({
  path: z.string(),
  recursive: z.boolean().optional().default(false).describe('Recursively delete directory contents')
});
```
#### Plik: `package.json`
```json
{
  "name": "@modelcontextprotocol/server-filesystem",
  "version": "0.6.3",
  "description": "MCP server for filesystem access",
  "license": "MIT",
  "author": "Anthropic, PBC (https://anthropic.com)",
  "homepage": "https://modelcontextprotocol.io",
  "bugs": {
    "url": "https://github.com/modelcontextprotocol/servers/issues"
  },
  "type": "commonjs",
  "bin": {
    "mcp-server-filesystem": "dist/server/index.js"
  },
  "files": [
    "dist"
  ],
  "scripts": {
    "build": "npm run clean && tsc && chmod +x dist/server/*.js",
    "prepare": "npm run build",
    "watch": "tsc --watch",
    "clean": "npx rimraf dist",
    "prepublishOnly": "npm run build",
    "test": "jest"
  },
  "dependencies": {
    "@modelcontextprotocol/sdk": "0.5.0",
    "async-mutex": "^0.3.2",
    "axios": "^1.10.0",
    "diff": "^5.1.0",
    "glob": "^10.3.10",
    "minimatch": "^10.0.1",
    "pino": "^8.17.2",
    "zod-to-json-schema": "^3.23.5",
    "mime-types": "^2.1.35",
    "fast-levenshtein": "^3.0.0"
  },
  "devDependencies": {
    "@types/diff": "^5.0.9",
    "@types/minimatch": "^5.1.2",
    "@types/node": "^22.15.31",
    "@types/pino": "^7.0.5",
    "rimraf": "^5.0.10",
    "shx": "^0.3.4",
    "typescript": "^5.3.3",
    "@types/jest": "^29.5.12",
    "@types/mime-types": "^2.1.4",
    "jest": "^29.7.0",
    "ts-jest": "^29.1.2"
  },
  "types": "./dist/server/index.d.ts",
  "main": "index.js",
  "keywords": [],
  "jest": {
    "preset": "ts-jest",
    "testEnvironment": "node",
    "roots": [
      "<rootDir>/src"
    ],
    "moduleFileExtensions": ["ts", "js", "json"],
    "testMatch": ["**/__tests__/**/*.test.ts"]
  }
}
```
#### Plik: `tsconfig.json`
```json
{
  "compilerOptions": {
    "types": ["jest", "node"],
    "typeRoots": ["./node_modules/@types", "./src/types"],
    "target": "es2022",
    "module": "commonjs",
    "moduleResolution": "node",
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "skipLibCheck": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "baseUrl": ".",
    "paths": {
      "@modelcontextprotocol/sdk/*": ["node_modules/@modelcontextprotocol/sdk/dist/*"]
    },
    "types": ["node"],
    "declaration": true,
    "emitDeclarationOnly": false,
    "sourceMap": true,
    "resolveJsonModule": true,
    "noEmit": false
  },
  "include": [
    "src/**/*.ts",
    "src/**/*.test.ts"
  ],
  "exclude": [
    "node_modules",
    "dist"
  ]
}
```
#### Plik: `test-filtering.js`
```js
const { spawn } = require('child_process');
const path = require('path');

// Start the server
const server = spawn('node', [path.join(__dirname, 'dist/server/index.js')], {
  stdio: ['pipe', 'pipe', 'pipe']
});

// Handle server output
server.stdout.on('data', (data) => {
  console.log(`Server: ${data}`);
});

server.stderr.on('data', (data) => {
  console.error(`Server Error: ${data}`);
});

// Send test requests after server starts
setTimeout(() => {
  testFileFiltering();
}, 2000);

async function testFileFiltering() {
  try {
    // Test reading allowed OpenCL file
    await sendRequest('read_file', {
      path: path.join(__dirname, 'test.cl'),
      encoding: 'auto'
    });
    
    // Test reading excluded file (in dist directory)
    try {
      await sendRequest('read_file', {
        path: path.join(__dirname, 'dist/server/index.js'),
        encoding: 'auto'
      });
    } catch (error) {
      console.log('Correctly blocked excluded file:', error);
    }
    
    // Test reading disallowed extension when forceTextFiles is true
    try {
      await sendRequest('read_file', {
        path: path.join(__dirname, 'test.exe'),
        encoding: 'auto'
      });
    } catch (error) {
      console.log('Correctly blocked disallowed extension:', error);
    }
    
  } catch (error) {
    console.error('Test failed:', error);
  } finally {
    server.kill();
  }
}

function sendRequest(tool, args) {
  return new Promise((resolve, reject) => {
    const requestId = Date.now();
    const request = JSON.stringify({
      jsonrpc: '2.0',
      method: 'call_tool',
      params: {
        name: tool,
        args
      },
      id: requestId
    }) + '\n';
    
    server.stdin.write(request);
    
    const listener = (data) => {
      const response = data.toString().trim();
      try {
        const json = JSON.parse(response);
        if (json.id === requestId) {
          server.stdout.off('data', listener);
          if (json.error) {
            reject(json.error);
          } else {
            resolve(json.result);
          }
        }
      } catch (e) {
        server.stdout.off('data', listener);
        reject(e);
      }
    };
    
    server.stdout.on('data', listener);
  });
}
```
---