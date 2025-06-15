# Projekt: gatto nero mcp no node and dist
## Katalog główny: `/home/lukasz/projects/gatto-ps-ai`
## Łączna liczba unikalnych plików: 16
---
## Grupa: gatto nero mcp no node and dist
**Opis:** kod gatto nerro mcp filesystem
**Liczba plików w grupie:** 16

### Lista plików:
- `index.ts`
- `src/utils/hintMap.ts`
- `src/utils/binaryDetect.ts`
- `src/utils/performance.ts`
- `src/utils/pathUtils.ts`
- `src/server/index.ts`
- `src/server/config.ts`
- `src/types/errors.ts`
- `src/core/security.ts`
- `src/core/fuzzyEdit.ts`
- `src/core/fileInfo.ts`
- `src/core/toolHandlers.ts`
- `src/core/schemas.ts`
- `package.json`
- `tsconfig.json`
- `test-filtering.js`

### Zawartość plików:
#### Plik: `index.ts`
```ts
// This file is intentionally left blank after refactoring to src/server/index.ts
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

import { loadConfig } from './config.js';
import { setupToolHandlers } from '../core/toolHandlers.js';
import * as schemas from '../core/schemas.js';
import { expandHome, normalizePath } from '../utils/pathUtils.js';

async function main() {
  const config = await loadConfig();

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

  const allowedDirectories = config.allowedDirectories.map(dir => normalizePath(path.resolve(expandHome(dir))));

  await Promise.all(allowedDirectories.map(async (dir) => {
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

  setupToolHandlers(server, allowedDirectories, logger, config);
// list_tools jest już zarejestrowane w toolHandlers.ts, więc usuwamy duplikat

  const transport = new StdioServerTransport();
  await server.connect(transport);

  logger.info({ version: '0.7.0', allowedDirectories, config }, 'Enhanced MCP Filesystem Server started');
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

export const ConfigSchema = z.object({
  allowedDirectories: z.array(z.string()),
  fileFiltering: z.object({
    defaultExcludes: z.array(z.string()).default([
      '**/build/**',
      '**/dist/**',
      '**/node_modules/**',
      '**/.git/**',
      '**/*.jpg', '**/*.png', '**/*.gif', '**/*.pdf',
      '**/*.zip', '**/*.tar', '**/*.gz'
    ]),
    allowedExtensions: z.array(z.string()).default([
      '*.txt', '*.js', '*.jsx', '*.ts', '*.tsx', '*.json', '*.yaml', '*.yml',
      '*.html', '*.htm', '*.css', '*.scss', '*.sass', '*.less', '*.py', '*.java', '*.go',
      '*.rs', '*.rb', '*.php', '*.sh', '*.bash', '*.zsh', '*.md', '*.markdown', '*.xml',
      '*.svg', '*.csv', '*.toml', '*.ini', '*.cfg', '*.conf', '*.env', '*.ejs', '*.pug',
      '*.vue', '*.svelte', '*.graphql', '*.gql', '*.proto', '*.kt', '*.kts', '*.swift',
      '*.m', '*.h', '*.c', '*.cpp', '*.hpp', '*.cs', '*.fs', '*.fsx', '*.clj', '*.cljs',
      '*.cljc', '*.edn', '*.ex', '*.exs', '*.erl', '*.hrl', '*.lua', '*.sql', '*.pl',
      '*.pm', '*.r', '*.jl', '*.dart', '*.groovy', '*.gradle', '*.nim', '*.zig', '*.v',
      '*.vh', '*.vhd', '*.cl', '*.tex', '*.sty', '*.cls', '*.rst', '*.adoc', '*.asciidoc'
    ]),
    forceTextFiles: z.boolean().default(true)
  }).default({}),
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
    maxConcurrentEdits: z.number().positive().default(10)
  }).default({})
});

export type Config = z.infer<typeof ConfigSchema>;

export async function loadConfig(): Promise<Config> {
    const args = process.argv.slice(2);
    if (args.length > 0 && (args[0] === '--config' || args[0] === '-c')) {
        if (args.length < 2) {
            console.error("Usage: mcp-server-filesystem --config <config-file>");
            console.error("   or: mcp-server-filesystem <allowed-directory> [additional-directories...]");
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
            defaultExcludes: [
                '**/build/**',
                '**/dist/**',
                '**/node_modules/**',
                '**/.git/**',
                '**/*.jpg', '**/*.png', '**/*.gif', '**/*.pdf',
                '**/*.zip', '**/*.tar', '**/*.gz'
            ],
            allowedExtensions: [
                '*.txt', '*.js', '*.jsx', '*.ts', '*.tsx', '*.json', '*.yaml', '*.yml',
                '*.html', '*.htm', '*.css', '*.scss', '*.sass', '*.less', '*.py', '*.java', '*.go',
                '*.rs', '*.rb', '*.php', '*.sh', '*.bash', '*.zsh', '*.md', '*.markdown', '*.xml',
                '*.svg', '*.csv', '*.toml', '*.ini', '*.cfg', '*.conf', '*.env', '*.ejs', '*.pug',
                '*.vue', '*.svelte', '*.graphql', '*.gql', '*.proto', '*.kt', '*.kts', '*.swift',
                '*.m', '*.h', '*.c', '*.cpp', '*.hpp', '*.cs', '*.fs', '*.fsx', '*.clj', '*.cljs',
                '*.cljc', '*.edn', '*.ex', '*.exs', '*.erl', '*.hrl', '*.lua', '*.sql', '*.pl',
                '*.pm', '*.r', '*.jl', '*.dart', '*.groovy', '*.gradle', '*.nim', '*.zig', '*.v',
                '*.vh', '*.vhd', '*.cl', '*.tex', '*.sty', '*.cls', '*.rst', '*.adoc', '*.asciidoc'
            ],
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
            maxConcurrentEdits: 10
        }
    });
}
```
#### Plik: `src/types/errors.ts`
```ts
import { HintInfo, HINTS } from "../utils/hintMap.js";

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
#### Plik: `src/core/fuzzyEdit.ts`
```ts
import fs from 'node:fs/promises';
import { createTwoFilesPatch } from 'diff';
import { isBinaryFile } from '../utils/binaryDetect.js';
import { createError } from '../types/errors.js';
import { get as fastLevenshtein } from 'fast-levenshtein';
import { PerformanceTimer } from '../utils/performance.js';
import type { Logger } from 'pino';
import type { EditOperation } from './schemas.js';
import type { Config } from '../server/config.js';

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

// Using fast-levenshtein, the optimized native JS version is no longer primary
// function levenshteinDistanceOptimized(str1: string, str2: string): number {
  if (str1 === str2) return 0;
  if (str1.length === 0) return str2.length;
  if (str2.length === 0) return str1.length;
  
  const shorter = str1.length <= str2.length ? str1 : str2;
  const longer = str1.length <= str2.length ? str2 : str1;
  
  let previousRow = Array(shorter.length + 1).fill(0).map((_, i) => i);
  
  for (let i = 0; i < longer.length; i++) {
    const currentRow = [i + 1];
    for (let j = 0; j < shorter.length; j++) {
      const cost = longer[i] === shorter[j] ? 0 : 1;
      currentRow.push(Math.min(
        currentRow[j] + 1,
        previousRow[j + 1] + 1,
        previousRow[j] + cost
      ));
    }
    previousRow = currentRow;
  }
  // return previousRow[shorter.length];
// }

function calculateSimilarity(distance: number, maxLength: number): number {
  return Math.max(0, 1 - (distance / maxLength));
}

function validateEdits(edits: Array<{oldText: string, newText: string}>, debug: boolean, logger: Logger): void {
  for (let i = 0; i < edits.length; i++) {
    for (let j = i + 1; j < edits.length; j++) {
      const edit1 = edits[i];
      const edit2 = edits[j];
      if (edit1.oldText.includes(edit2.oldText) || edit2.oldText.includes(edit1.oldText)) {
        const warning = `Warning: Potentially overlapping oldText in edits ${i+1} and ${j+1}`;
        logger.warn({ edit1Index: i, edit2Index: j }, warning);
      }
      if (edit1.newText.includes(edit2.oldText) || edit2.newText.includes(edit1.oldText)) {
        const warning = `Warning: newText in edit ${i+1} contains oldText from edit ${j+1} - potential mutual overlap`;
        logger.warn({ edit1Index: i, edit2Index: j }, warning);
      }
    }
  }
}

function applyRelativeIndentation(
  newLines: string[], 
  oldLines: string[], 
  originalIndent: string,
  preserveMode: 'auto' | 'strict' | 'normalize'
): string[] {
  switch (preserveMode) {
    case 'strict':
      return newLines.map(line => originalIndent + line.trimStart());
    case 'normalize':
      return newLines.map(line => originalIndent + line.trimStart());
    case 'auto':
    default:
      return newLines.map((line, idx) => {
        if (idx === 0) {
          return originalIndent + line.trimStart();
        }
        const oldLineIndex = Math.min(idx, oldLines.length - 1);
        const newLineIndent = line.match(/^\s*/)?.[0] || '';
        const baseOldIndent = oldLines[0]?.match(/^\s*/)?.[0]?.length || 0;
        const relativeIndentChange = newLineIndent.length - baseOldIndent;
        const finalIndent = originalIndent + ' '.repeat(Math.max(0, relativeIndentChange));
        return finalIndent + line.trimStart();
      });
  }
}

function getContextLines(text: string, lineNumber: number, contextSize: number): string {
  const lines = text.split('\n');
  const start = Math.max(0, lineNumber - contextSize);
  const end = Math.min(lines.length, lineNumber + contextSize + 1);
  // Return actual line numbers, so add 1 to start index for display if lines are 1-indexed in user's mind
  return lines.slice(start, end).map((line, i) => `${start + i + 1}: ${line}`).join('\n');
}

export async function applyFileEdits(
  filePath: string,
  edits: EditOperation[], // Updated to use EditOperation type
  config: FuzzyMatchConfig,
  logger: Logger,
  globalConfig: Config
): Promise<ApplyFileEditsResult> {
  const timer = new PerformanceTimer('applyFileEdits', logger, globalConfig);
  let levenshteinIterations = 0;
  const appliedRanges: AppliedEditRange[] = [];
  
  try {
    const buffer = await fs.readFile(filePath);
    if (isBinaryFile(buffer, filePath)) {
      throw createError(
        'BINARY_FILE_ERROR',
        'Cannot edit binary files',
        { filePath, detectedAs: 'binary' }
      );
    }

    const originalContent = normalizeLineEndings(buffer.toString('utf-8'));
    let modifiedContent = originalContent;

    validateEdits(edits, config.debug, logger);

    for (const [editIndex, edit] of edits.entries()) {
      const normalizedOld = normalizeLineEndings(edit.oldText);
      const normalizedNew = normalizeLineEndings(edit.newText);
      let matchFound = false;

      const exactMatchIndex = modifiedContent.indexOf(normalizedOld);
      if (exactMatchIndex !== -1) {
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
          charCount += contentLines[i].length + 1; // +1 for newline
        }
        const originalIndent = contentLines[lineNumberOfMatch].match(/^\s*/)?.[0] || '';

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
        matchFound = true;
        appliedRanges.push({
          startLine: currentEditTargetRange.startLine,
          endLine: currentEditTargetRange.startLine + indentedNewLines.length - 1,
          editIndex
        });
      } else {
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

        const windowSizes = [
          oldLines.length,
          Math.max(1, oldLines.length - 1),
          oldLines.length + 1,
          Math.max(1, oldLines.length - 2),
          oldLines.length + 2
        ].filter((size, index, arr) => arr.indexOf(size) === index && size > 0);

        for (const windowSize of windowSizes) {
          if (windowSize > contentLines.length) continue;

          for (let i = 0; i <= contentLines.length - windowSize; i++) {
            const windowLines = contentLines.slice(i, i + windowSize);
            const windowText = windowLines.join('\n');
            const processedWindow = preprocessText(windowText, config);
            
            levenshteinIterations++;
            // Use fast-levenshtein
            const distance = fastLevenshtein(processedOld, processedWindow);
            const similarity = calculateSimilarity(distance, Math.max(processedOld.length, processedWindow.length));

            if (distance < bestMatch.distance) {
              bestMatch = { distance, index: i, text: windowText, similarity, windowSize };
            }
            if (distance === 0) break;
          }
          if (bestMatch.distance === 0) break;
        }

        const distanceThreshold = Math.floor(processedOld.length * config.maxDistanceRatio);

        if (bestMatch.index !== -1 && 
            bestMatch.distance <= distanceThreshold && 
            bestMatch.similarity >= config.minSimilarity) {
          
          const newLines = normalizedNew.split('\n');
          const originalIndent = contentLines[bestMatch.index].match(/^\s*/)?.[0] || '';
          const indentedNewLines = applyRelativeIndentation(
            newLines, 
            oldLines, 
            originalIndent, 
            config.preserveLeadingWhitespace
          );

          contentLines.splice(bestMatch.index, bestMatch.windowSize, ...indentedNewLines);
          modifiedContent = contentLines.join('\n');
          matchFound = true;
        } else if (bestMatch.similarity >= 0.5) {
          if (edit.forcePartialMatch) {
            logger.warn(`Applying forced partial match for edit ${editIndex + 1} (similarity: ${bestMatch.similarity.toFixed(3)}) due to 'forcePartialMatch: true'.`);
            const newLines = normalizedNew.split('\n');
            const originalIndent = contentLines[bestMatch.index].match(/^\s*/)?.[0] || '';
            const oldLinesForIndent = normalizedOld.split('\n');
            const indentedNewLines = applyRelativeIndentation(
              newLines, 
              oldLinesForIndent,
              originalIndent, 
              config.preserveLeadingWhitespace
            );
            const currentEditTargetRange = {
              startLine: bestMatch.index,
              endLine: bestMatch.index + bestMatch.windowSize - 1
            };

            for (const appliedRange of appliedRanges) {
              if (doRangesOverlap(appliedRange, currentEditTargetRange)) {
                throw createError(
                  'OVERLAPPING_EDIT',
                  `Edit ${editIndex + 1} (fuzzy match at line ${bestMatch.index + 1}) overlaps with previously applied edit ${appliedRange.editIndex + 1}. ` +
                  `Current edit targets lines ${currentEditTargetRange.startLine + 1}-${currentEditTargetRange.endLine + 1}. ` +
                  `Previous edit affected lines ${appliedRange.startLine + 1}-${appliedRange.endLine + 1}.`,
                  {
                    conflictingEditIndex: editIndex,
                    previousEditIndex: appliedRange.editIndex,
                    currentEditTargetRange,
                    previousEditAffectedRange: appliedRange,
                    similarity: bestMatch.similarity
                  }
                );
              }
            }

            contentLines.splice(bestMatch.index, bestMatch.windowSize, ...indentedNewLines);
            modifiedContent = contentLines.join('\n');
            matchFound = true;
            appliedRanges.push({
              startLine: currentEditTargetRange.startLine,
              endLine: currentEditTargetRange.startLine + indentedNewLines.length - 1,
              editIndex
            });
          } else {
            const contextText = getContextLines(normalizeLineEndings(originalContent), bestMatch.index, 3);
            const partialDiff = createUnifiedDiff(bestMatch.text, processedOld, filePath);
            throw createError(
              'PARTIAL_MATCH',
              `Partial match found for edit ${editIndex + 1} (similarity: ${bestMatch.similarity.toFixed(3)}).` +
              `\n=== Context (around line ${bestMatch.index + 1} in preprocessed content) ===\n${contextText}\n` +
              `\n=== Diff (actual found text vs. your preprocessed oldText) ===\n${partialDiff}\n` +
              `\n=== Suggested Fix ===\n` +
              `1. Adjust 'oldText' to match the content more closely.\n` +
              `2. Or set 'forcePartialMatch: true' for this edit operation if this partial match is acceptable.`, 
              {
                editIndex,
                similarity: bestMatch.similarity,
                bestMatchPreview: bestMatch.text.substring(0, 100),
                context: contextText,
                diff: partialDiff
              }
            );
          }
        }
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
          // Match exclude patterns against the relative path, case-insensitively
          const shouldExclude = excludePatterns.some(p => minimatch(relativePath, p, { dot: true, nocase: true }));

          if (shouldExclude) {
            continue;
          }

          // Check if path should be excluded based on file filtering rules
          if (config.fileFiltering.defaultExcludes.some(p => minimatch(relativePath, p, { dot: true }))) {
            continue;
          }

          // Check allowed extensions if forceTextFiles is true
          if (config.fileFiltering.forceTextFiles) {
            const ext = path.extname(fullPath).toLowerCase();
            if (!config.fileFiltering.allowedExtensions.some(p => minimatch(`*${ext}`, p))) {
              continue;
            }
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
        
        // Skip excluded paths and disallowed extensions
        const relativePath = path.relative(basePath, childPath);
        if (config.fileFiltering.defaultExcludes.some(p => minimatch(relativePath, p, { dot: true }))) {
          continue;
        }
        
        // Check allowed extensions if forceTextFiles is true
        if (dirent.isFile() && config.fileFiltering.forceTextFiles) {
          const ext = path.extname(childPath).toLowerCase();
          if (!config.fileFiltering.allowedExtensions.some(p => minimatch(`*${ext}`, p))) {
            continue;
          }
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
import fs from 'node:fs/promises';
import path from 'node:path';
import { minimatch } from 'minimatch';

import { createError, StructuredError } from '../types/errors.js';
import { PerformanceTimer } from '../utils/performance.js';
import { isBinaryFile } from '../utils/binaryDetect.js';
import { validatePath } from './security.js';
import { applyFileEdits, FuzzyMatchConfig } from './fuzzyEdit.js';
import { getFileStats, searchFiles, readMultipleFilesContent, FileReadResult, getDirectoryTree } from './fileInfo.js'; 
import * as schemas from './schemas.js';
// Import specific types that were causing issues if not directly imported
import type { ListDirectoryEntry, DirectoryTreeEntry, EditOperation } from './schemas.js'; 

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

function shouldSkipPath(filePath: string, config: Config): boolean {
  const relativePath = path.relative(config.allowedDirectories[0], filePath);
  
  // Check against default excludes
  for (const pattern of config.fileFiltering.defaultExcludes) {
    if (minimatch(relativePath, pattern)) {
      return true;
    }
  }
  
  // Check allowed extensions if forceTextFiles is true
  if (config.fileFiltering.forceTextFiles) {
    const ext = path.extname(filePath).toLowerCase();
    if (!config.fileFiltering.allowedExtensions.includes(`*${ext}`)) {
      return true;
    }
  }
  
  return false;
}

export function setupToolHandlers(server: Server, allowedDirectories: string[], logger: Logger, config: Config) {
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
          return { result: allowedDirectories };
        }

        case 'read_file': {
          const parsed = schemas.ReadFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ReadFileArgsSchema) });
          const timer = new PerformanceTimer('read_file_handler', logger, config);
          try {
            const validatedPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
            
            // Check if path should be excluded based on file filtering rules
            const relativePath = path.relative(config.allowedDirectories[0], validatedPath);
            if (config.fileFiltering.defaultExcludes.some(p => minimatch(relativePath, p, { dot: true }))) {
              throw createError('ACCESS_DENIED', 'Path matches excluded pattern');
            }
            
            // Check allowed extensions if forceTextFiles is true
            if (config.fileFiltering.forceTextFiles) {
              const ext = path.extname(validatedPath).toLowerCase();
              if (!config.fileFiltering.allowedExtensions.some(p => minimatch(`*${ext}`, p))) {
                throw createError('ACCESS_DENIED', 'File extension not allowed');
              }
            }
            
            const rawBuffer = await fs.readFile(validatedPath);
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

            // const timer = new PerformanceTimer('read_multiple_files_handler', logger, config); // Timer is now within readMultipleFilesContent
            const fileReadResults = await readMultipleFilesContent(
              parsed.data.paths,
              parsed.data.encoding,
              allowedDirectories,
              logger,
              config
            );
            // timer.end(...); // Logging for timer is handled within readMultipleFilesContent
            return { result: fileReadResults };
        }

        case 'write_file': {
          const parsed = schemas.WriteFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.WriteFileArgsSchema) });
          const timer = new PerformanceTimer('write_file_handler', logger, config);
          try {
            const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config); // isWriteOperation = true
            
            // Check if path should be excluded based on file filtering rules
            const relativePath = path.relative(config.allowedDirectories[0], validPath);
            if (config.fileFiltering.defaultExcludes.some(p => minimatch(relativePath, p, { dot: true }))) {
              throw createError('ACCESS_DENIED', 'Path matches excluded pattern');
            }
            
            // Check allowed extensions if forceTextFiles is true
            if (config.fileFiltering.forceTextFiles) {
              const ext = path.extname(validPath).toLowerCase();
              if (!config.fileFiltering.allowedExtensions.some(p => minimatch(`*${ext}`, p))) {
                throw createError('ACCESS_DENIED', 'File extension not allowed');
              }
            }
            
            const lock = getFileLock(validPath, config, requestLogger);
            await lock.runExclusive(async () => {
              const contentBuffer = Buffer.from(parsed.data.content, parsed.data.encoding);
              await fs.writeFile(validPath, contentBuffer);
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
            throw createError('FILE_FILTERED_OUT', 'File is excluded by filtering rules', { path: validPath });
          }
          const fuzzyConfig: FuzzyMatchConfig = {
            maxDistanceRatio: config.fuzzyMatching.maxDistanceRatio,
            minSimilarity: config.fuzzyMatching.minSimilarity,
            caseSensitive: config.fuzzyMatching.caseSensitive,
            ignoreWhitespace: config.fuzzyMatching.ignoreWhitespace,
            preserveLeadingWhitespace: config.fuzzyMatching.preserveLeadingWhitespace,
            debug: config.logging.level === 'debug',
          };

          const lock = getFileLock(validPath, config);
          const { modifiedContent, formattedDiff } = await lock.runExclusive(async () => {
            const editResult = await applyFileEdits(
              validPath, // Use validated path
              parsed.data.edits,
              fuzzyConfig,
              logger,
              config
            );
            return editResult;
          });

          const responseText = parsed.data.dryRun 
            ? `Dry run: File '${parsed.data.path}' would be modified. Diff:\n${formattedDiff}`
            : `File '${parsed.data.path}' edited successfully. Diff:\n${formattedDiff}`;
          
          return { content: [{ type: 'text', text: responseText }] };
        }
        
        case 'create_directory': {
          const parsed = schemas.CreateDirectoryArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.CreateDirectoryArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config); // isWriteOperation = true
          await fs.mkdir(validPath, { recursive: true });
          return { content: [{ type: 'text', text: `Directory created: ${parsed.data.path}` }] };
        }

        case 'list_directory': {
          const parsed = schemas.ListDirectoryArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ListDirectoryArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          const entries = await fs.readdir(validPath, { withFileTypes: true });
          const results: ListDirectoryEntry[] = [];
          for (const dirent of entries) {
            let type: ListDirectoryEntry['type'] = 'other';
            if (dirent.isFile()) type = 'file';
            else if (dirent.isDirectory()) type = 'directory';
            else if (dirent.isSymbolicLink()) type = 'symlink';
            
            const entryPath = path.join(validPath, dirent.name);
            let size: number | undefined = undefined;
            if (type === 'file') {
                try {
                    const stats = await fs.stat(entryPath);
                    size = stats.size;
                } catch (statError) {
                    logger.warn({ path: entryPath, error: statError }, 'Failed to get stats for file in list_directory');
                }
            }
            if (!shouldSkipPath(entryPath, config)) {
              results.push({ name: dirent.name, path: path.relative(config.allowedDirectories[0], entryPath).replace(/\\/g, '/'), type, size });
            }
          }
          return { result: results };
        }

        case 'move_file': {
          const parsed = schemas.MoveFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.MoveFileArgsSchema) });
          const validSource = await validatePath(parsed.data.source, allowedDirectories, logger, config); // isWriteOperation = true (on source)
          const validDestination = await validatePath(parsed.data.destination, allowedDirectories, logger, config); // isWriteOperation = true (on destination)
          if (shouldSkipPath(validSource, config)) {
            throw createError('FILE_FILTERED_OUT', 'File is excluded by filtering rules', { path: validSource });
          }
          if (shouldSkipPath(validDestination, config)) {
            throw createError('FILE_FILTERED_OUT', 'File is excluded by filtering rules', { path: validDestination });
          }
          
          const sourceLock = getFileLock(validSource, config);
          const destLock = getFileLock(validDestination, config); // Potentially lock destination too if it might exist or be created
          
          await sourceLock.runExclusive(async () => {
            // If destination is different, also acquire its lock if not already held
            if (validSource !== validDestination && !destLock.isLocked()) {
              await destLock.runExclusive(async () => {
                await fs.rename(validSource, validDestination);
              });
            } else {
              // If source and destination are same or destLock already acquired (e.g. by sourceLock if paths are same)
              await fs.rename(validSource, validDestination);
            }
          });
          return { content: [{ type: 'text', text: `Moved from ${parsed.data.source} to ${parsed.data.destination}` }] };
        }

        case 'delete_file': {
          const parsed = schemas.DeleteFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.DeleteFileArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config); // isWriteOperation = true
          if (shouldSkipPath(validPath, config)) {
            throw createError('FILE_FILTERED_OUT', 'File is excluded by filtering rules', { path: validPath });
          }
          const lock = getFileLock(validPath, config);
          await lock.runExclusive(async () => {
            await fs.unlink(validPath);
          });
          return { content: [{ type: 'text', text: `File deleted: ${parsed.data.path}` }] };
        }

        case 'delete_directory': {
          const parsed = schemas.DeleteDirectoryArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.DeleteDirectoryArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config); // isWriteOperation = true
          if (shouldSkipPath(validPath, config)) {
            throw createError('FILE_FILTERED_OUT', 'File is excluded by filtering rules', { path: validPath });
          }
          const lock = getFileLock(validPath, config); // Lock the directory itself
          await lock.runExclusive(async () => {
            await fs.rm(validPath, { recursive: parsed.data.recursive || false, force: false }); // force: false for safety
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
              parsed.data.useExactPatterns || false
            );
            
            if (parsed.data.maxResults && results.length > parsed.data.maxResults) {
              results.length = parsed.data.maxResults;
            }
            
            timer.end({ resultsCount: results.length });
            return { results };
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
            throw createError('FILE_FILTERED_OUT', 'File is excluded by filtering rules', { path: validPath });
          }
          const stats = await getFileStats(validPath, logger, config);
          return { result: stats };
        }

        case 'directory_tree': {
            const parsed = schemas.DirectoryTreeArgsSchema.safeParse(request.params.args);
            if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.DirectoryTreeArgsSchema) });
            const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
            const tree = await getDirectoryTree(validPath, allowedDirectories, logger, config);
            logger.debug({ tree }, 'Generated directory tree (getDirectoryTree)');
            return { content: [{ type: 'text', text: JSON.stringify(tree, null, 2) }] };
        }

        case 'server_stats': {
            const parsed = schemas.ServerStatsArgsSchema.safeParse(request.params.args);
            if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ServerStatsArgsSchema) });
            const stats = { requestCount, editOperationCount, config };
            return { content: [{ type: 'text', text: JSON.stringify(stats, null, 2) }] };
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

export const DirectoryTreeArgsSchema = z.object({
  path: z.string(),
});

// Define the recursive DirectoryTreeEntrySchema
// We need to use z.lazy to handle recursive types with Zod
export const DirectoryTreeEntrySchema: z.ZodType<DirectoryTreeEntry> = z.lazy(() =>
  z.object({
    name: z.string().describe('Name of the file or directory'),
    path: z.string().describe('Full absolute path of the file or directory'),
    type: z.enum(['file', 'directory']).describe('Type of the entry'),
    children: z.array(DirectoryTreeEntrySchema).optional().describe('Children of the directory entry, undefined for files'),
    // We might want to add size for files or other metadata later
    // size: z.number().optional().describe('Size of the file in bytes, undefined for directories'), 
  })
);

// Define the TypeScript interface for DirectoryTreeEntry for clarity
export interface DirectoryTreeEntry {
  name: string;
  path: string;
  type: 'file' | 'directory';
  children?: DirectoryTreeEntry[];
  // size?: number;
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
  "keywords": []
}
```
#### Plik: `tsconfig.json`
```json
{
  "compilerOptions": {
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
    "src/**/*.ts"
  ],
  "exclude": [
    "node_modules",
    "dist",
    "**/*.test.ts",
    "**/*.spec.ts"
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