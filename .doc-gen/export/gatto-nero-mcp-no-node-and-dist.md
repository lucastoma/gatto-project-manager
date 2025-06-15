# Projekt: gatto nero mcp no node and dist
## Katalog główny: `D:\projects\gatto-ps-ai-link1`
## Łączna liczba unikalnych plików: 15
---
## Grupa: gatto nero mcp no node and dist
**Opis:** kod gatto nerro mcp filesystem
**Liczba plików w grupie:** 15

### Lista plików:
- `index.ts`
- `package.json`
- `src/core/fileInfo.ts`
- `src/core/fuzzyEdit.ts`
- `src/core/schemas.ts`
- `src/core/security.ts`
- `src/core/toolHandlers.ts`
- `src/server/config.ts`
- `src/server/index.ts`
- `src/types/errors.ts`
- `src/utils/binaryDetect.ts`
- `src/utils/hintMap.ts`
- `src/utils/pathUtils.ts`
- `src/utils/performance.ts`
- `tsconfig.json`

### Zawartość plików:
#### Plik: `index.ts`
```ts
// This file is intentionally left blank after refactoring to src/server/index.ts
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
  "bugs": "https://github.com/modelcontextprotocol/servers/issues",
  "type": "module",
  "bin": {
    "mcp-server-filesystem": "dist/server/index.js"
  },
  "files": [
    "dist"
  ],
  "scripts": {
    "build": "npm run clean && tsc && shx chmod +x dist/server/*.js",
    "prepare": "npm run build",
    "watch": "tsc --watch",
    "clean": "rimraf dist",
    "prepublishOnly": "npm run build"
  },
  "dependencies": {
    "async-mutex": "^0.3.2",
    "pino": "^8.17.2",
    "@modelcontextprotocol/sdk": "0.5.0",
    "diff": "^5.1.0",
    "glob": "^10.3.10",
    "minimatch": "^10.0.1",
    "zod-to-json-schema": "^3.23.5"
  },
  "devDependencies": {
    "@types/pino": "^7.0.5",
    "rimraf": "^5.0.5",
    "@types/diff": "^5.0.9",
    "@types/minimatch": "^5.1.2",
    "@types/node": "^22",
    "shx": "^0.3.4",
    "typescript": "^5.3.3"
  },
  "types": "./dist/server/index.d.ts"
}
```
#### Plik: `src/core/fileInfo.ts`
```ts
import fs from 'node:fs/promises';
import path from 'node:path';
import { minimatch } from 'minimatch';
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

export async function getFileStats(filePath: string, logger: Logger, config: Config): Promise<FileInfo> {
  const timer = new PerformanceTimer('getFileStats', logger, config);
  
  try {
    const stats = await fs.stat(filePath);
    const buffer = await fs.readFile(filePath);
    const isBinary = isBinaryFile(buffer, filePath);
    
    const result = {
      size: stats.size,
      created: stats.birthtime,
      modified: stats.mtime,
      accessed: stats.atime,
      isDirectory: stats.isDirectory(),
      isFile: stats.isFile(),
      permissions: stats.mode.toString(8).slice(-3),
      isBinary,
      mimeType: isBinary ? 'application/octet-stream' : 'text/plain'
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
  useExactPatterns: boolean = false
): Promise<string[]> {
  const timer = new PerformanceTimer('searchFiles', logger, config);
  const results: string[] = [];
  const visitedForThisSearch = new Set<string>();

  async function search(currentPath: string): Promise<void> {
    try {
      const stats = await fs.stat(currentPath);
      const inodeKey = `${stats.dev}-${stats.ino}`;
      
      if (visitedForThisSearch.has(inodeKey)) {
        logger.debug({ path: currentPath }, 'Skipping already visited inode (symlink loop)');
        return;
      }
      visitedForThisSearch.add(inodeKey);

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
            await search(fullPath);
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

  await search(rootPath);
  timer.end({ resultsCount: results.length });
  return results;
}

import type { DirectoryTreeEntry } from './schemas.js';

export async function getDirectoryTree(
  basePath: string,
  allowedDirectories: string[],
  logger: Logger,
  config: Config,
  currentDepth: number = 0, // Keep track of current depth
  maxDepth: number = -1 // Default to no max depth (-1 or undefined)
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
      let encodingUsed: 'utf-8' | 'base64' = 'utf-8'; // Default to utf-8

      if (requestedEncoding === 'base64') {
        content = rawBuffer.toString('base64');
        encodingUsed = 'base64';
      } else if (requestedEncoding === 'auto') {
        if (isBinaryFile(rawBuffer, validPath)) {
          content = rawBuffer.toString('base64');
          encodingUsed = 'base64';
        } else {
          content = rawBuffer.toString('utf-8');
          encodingUsed = 'utf-8';
        }
      } else { // utf-8
        content = rawBuffer.toString('utf-8');
        encodingUsed = 'utf-8';
      }
      results.push({ path: filePath, content, encoding: encodingUsed }); // Use 'encoding'
    } catch (error: any) {
      logger.warn({ path: filePath, error: error.message }, 'Failed to read one of the files in readMultipleFilesContent');
      results.push({ path: filePath, content: `Error: ${error.message}`, encoding: 'error' }); // Error in content, encoding='error'
    }
  }

  timer.end({ filesCount: filePaths.length, resultsCount: results.length });
  return results;
}
```
#### Plik: `src/core/fuzzyEdit.ts`
```ts
import fs from 'node:fs/promises';
import { createTwoFilesPatch } from 'diff';
import { isBinaryFile } from '../utils/binaryDetect.js';
import { createError } from '../types/errors.js';
import { PerformanceTimer } from '../utils/performance.js';
import type { Logger } from 'pino';
import type { Config } from '../server/config.js';

export interface FuzzyMatchConfig {
  maxDistanceRatio: number;
  minSimilarity: number;
  caseSensitive: boolean;
  ignoreWhitespace: boolean;
  preserveLeadingWhitespace: 'auto' | 'strict' | 'normalize';
  debug: boolean;
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

function levenshteinDistanceOptimized(str1: string, str2: string): number {
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
  return previousRow[shorter.length];
}

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

export async function applyFileEdits(
  filePath: string,
  edits: Array<{oldText: string, newText: string}>,
  config: FuzzyMatchConfig,
  logger: Logger,
  globalConfig: Config
): Promise<ApplyFileEditsResult> {
  const timer = new PerformanceTimer('applyFileEdits', logger, globalConfig);
  let levenshteinIterations = 0;
  
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
        modifiedContent = modifiedContent.substring(0, exactMatchIndex) +
                          normalizedNew +
                          modifiedContent.substring(exactMatchIndex + normalizedOld.length);
        matchFound = true;
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
            const distance = levenshteinDistanceOptimized(processedOld, processedWindow);
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
            throw createError(
                'PARTIAL_MATCH',
                `Found a partial match for edit ${editIndex + 1} with similarity ${bestMatch.similarity.toFixed(3)}. Please adjust 'oldText' or matching parameters.`,
                { 
                    editIndex, 
                    similarity: bestMatch.similarity, 
                    bestMatchPreview: bestMatch.text.substring(0, 100) 
                }
            );
        }
      }

      if (!matchFound) {
        let errorMessage = `Could not find a close match for edit ${editIndex + 1}:\n---\n${edit.oldText}\n---`;
        throw createError(
          'FUZZY_MATCH_FAILED',
          errorMessage,
          { editIndex, levenshteinIterations }
        );
      }
    }

    const diff = createUnifiedDiff(originalContent, modifiedContent, filePath);
    const formattedDiff = "```diff\n" + diff + "\n```\n\n";

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

export const EditOperation = z.object({
  oldText: z.string().describe('Text to search for - can be slightly inaccurate'),
  newText: z.string().describe('Text to replace with')
});
export type EditOperation = z.infer<typeof EditOperation>;

export const getEditFileArgsSchema = (config: Config) => z.object({
  path: z.string(),
  edits: z.array(EditOperation),
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
    // If the path is relative, resolve it against the FIRST allowed directory instead of the server CWD.
    // This makes JSON-RPC requests intuitive: a client can pass "./" or "sub/dir" and it will be treated as relative
    // to the allowed directory supplied at server start (e.g. "mcp-test").
    const absolute = path.isAbsolute(expandedPath)
      ? path.resolve(expandedPath)
      : path.resolve(allowedDirectories[0] ?? process.cwd(), expandedPath);

    const normalizedRequested = normalizePath(absolute);

    const isAllowed = allowedDirectories.some(dir => {
      const relativePath = path.relative(dir, normalizedRequested);
      return !relativePath.startsWith('..' + path.sep) && relativePath !== '..';
    });
    
    if (!isAllowed) {
      throw createError(
        'ACCESS_DENIED',
        `Path outside allowed directories: ${absolute}`,
        { 
          requestedPath: absolute, 
          allowedDirectories: allowedDirectories 
        }
      );
    }

    try {
      const realPath = await fs.realpath(absolute);
      const normalizedReal = normalizePath(realPath);
      const isRealPathAllowed = allowedDirectories.some(dir => {
        const relativePath = path.relative(dir, normalizedReal);
        return !relativePath.startsWith('..' + path.sep) && relativePath !== '..';
      });
      
      if (!isRealPathAllowed) {
        throw createError(
          'ACCESS_DENIED',
          'Symlink target outside allowed directories',
          { symlinkTarget: realPath }
        );
      }
      
      timer.end({ result: 'success', realPath });
      return realPath;
    } catch (error) {
      // realpath may fail on Windows for non-existent or locked files/directories.
      // If the absolute path itself exists (lstat succeeds) and is inside allowedDirectories,
      // we can safely allow it.
      try {
        await fs.lstat(absolute);
        timer.end({ result: 'success', realPath: absolute });
        return absolute;
      } catch {
        const parentDir = path.dirname(absolute);
        try {
          const normalizedParent = normalizePath(parentDir); // parentDir is already absolute here
        const isParentAllowed = allowedDirectories.some(allowedDir => {
          const relativePath = path.relative(allowedDir, normalizedParent);
          // Check if normalizedParent is the same as allowedDir or a subdirectory
          return !relativePath.startsWith('..' + path.sep) && relativePath !== '..';
        });
        
        if (!isParentAllowed) {
          throw createError(
            'ACCESS_DENIED',
            'Parent directory outside allowed directories',
            { parentDirectory: normalizedParent } // Use normalizedParent for error reporting
          );
        }
        
        timer.end({ result: 'success', newFile: true });
        return absolute;
      } catch (parentError) {
        if ((parentError as any).code === 'ACCESS_DENIED') throw parentError; // Re-throw our custom error
        throw createError(
          'ACCESS_DENIED',
          `Parent directory does not exist or is not accessible: ${parentDir}`,
          { parentDirectory: parentDir }
        );
      }
      }
    }
  } catch (error) {
    timer.end({ result: 'error' });
    if ((error as any).code) {
      throw error;
    }
    throw createError('VALIDATION_ERROR', (error as Error).message || String(error));
  }
}
```
#### Plik: `src/core/toolHandlers.ts`
```ts
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { Mutex } from 'async-mutex';
import fs from 'node:fs/promises';
import path from 'node:path';

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

function getFileLock(filePath: string, config: Config): Mutex {
  if (!fileLocks.has(filePath)) {
    if (fileLocks.size >= config.concurrency.maxConcurrentEdits) {
      const oldestKey = fileLocks.keys().next().value;
      if (oldestKey) fileLocks.delete(oldestKey);
    }
    fileLocks.set(filePath, new Mutex());
  }
  return fileLocks.get(filePath)!;
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
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          const rawBuffer = await fs.readFile(validPath);
          let content: string;
          let encodingUsed: 'utf-8' | 'base64' = 'utf-8';
          const isBinary = isBinaryFile(rawBuffer, validPath);

          if (parsed.data.encoding === 'base64' || (parsed.data.encoding === 'auto' && isBinary)) {
            content = rawBuffer.toString('base64');
            encodingUsed = 'base64';
          } else {
            content = rawBuffer.toString('utf-8'); // Default to utf-8
          }
          return { result: { content, encoding: encodingUsed } };
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
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config); // isWriteOperation = true

          const lock = getFileLock(validPath, config);
          await lock.runExclusive(async () => {
            const contentBuffer = Buffer.from(parsed.data.content, parsed.data.encoding);
            await fs.writeFile(validPath, contentBuffer);
          });
          return { content: [{ type: 'text', text: `File written: ${parsed.data.path}` }] };
        }

        case 'edit_file': {
          editOperationCount++;
          const parsed = EditFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments for edit_file', { error: parsed.error, schema: zodToJsonSchema(EditFileArgsSchema) });
          
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config); // isWriteOperation = true
          const fileBuffer = await fs.readFile(validPath);
          if (isBinaryFile(fileBuffer, validPath)) { // Check if file is binary by extension or content
            throw createError('BINARY_FILE_EDIT', `Cannot edit binary file: ${validPath}`);
          }

          const fuzzyConfig: FuzzyMatchConfig = {
            maxDistanceRatio: parsed.data.maxDistanceRatio,
            minSimilarity: parsed.data.minSimilarity,
            caseSensitive: parsed.data.caseSensitive,
            ignoreWhitespace: parsed.data.ignoreWhitespace,
            preserveLeadingWhitespace: parsed.data.preserveLeadingWhitespace,
            debug: parsed.data.debug
          };
          
          let modifiedContent: string = '';
          let formattedDiff: string = '';

          const lock = getFileLock(validPath, config);
          await lock.runExclusive(async () => {
            const currentContent = await fs.readFile(validPath, 'utf-8');
            const editResult = await applyFileEdits(currentContent, parsed.data.edits as EditOperation[], fuzzyConfig, logger, config);
            modifiedContent = editResult.modifiedContent;
            formattedDiff = editResult.formattedDiff;
            if (!parsed.data.dryRun) {
              await fs.writeFile(validPath, modifiedContent, 'utf-8');
            }
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
            results.push({ name: dirent.name, path: path.relative(config.allowedDirectories[0], entryPath).replace(/\\/g, '/'), type, size });
          }
          return { result: results };
        }

        case 'move_file': {
          const parsed = schemas.MoveFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.MoveFileArgsSchema) });
          const validSource = await validatePath(parsed.data.source, allowedDirectories, logger, config); // isWriteOperation = true (on source)
          const validDestination = await validatePath(parsed.data.destination, allowedDirectories, logger, config); // isWriteOperation = true (on destination)
          
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
          
          const lock = getFileLock(validPath, config); // Lock the directory itself
          await lock.runExclusive(async () => {
            await fs.rm(validPath, { recursive: parsed.data.recursive || false, force: false }); // force: false for safety
          });
          return { content: [{ type: 'text', text: `Directory deleted: ${parsed.data.path}` }] };
        }

        case 'search_files': {
          const parsed = schemas.SearchFilesArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.SearchFilesArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          const results = await searchFiles(validPath, parsed.data.pattern, logger, config, parsed.data.excludePatterns || [], false);
          return { result: results };
        }

        case 'get_file_info': {
          const parsed = schemas.GetFileInfoArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.GetFileInfoArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
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
#### Plik: `src/server/config.ts`
```ts
import { z } from "zod";
import fs from "node:fs/promises";
import path from "node:path";

export const ConfigSchema = z.object({
  allowedDirectories: z.array(z.string()),
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
        
        try {
            const configPath = path.resolve(args[1]);
            const configContent = await fs.readFile(configPath, 'utf-8');
            const rawConfig = JSON.parse(configContent);
            return ConfigSchema.parse(rawConfig);
        } catch (error) {
            console.error("Error loading config file:", error);
            process.exit(1);
        }
    } else {
        if (args.length === 0) {
            console.error("Usage: mcp-server-filesystem --config <config-file>");
            console.error("   or: mcp-server-filesystem <allowed-directory> [additional-directories...]");
            process.exit(1);
        }
        
        const DEFAULT_MAX_DISTANCE_RATIO = parseFloat(process.env.MCP_EDIT_MAX_DISTANCE_RATIO || '0.25');
        const DEFAULT_MIN_SIMILARITY = parseFloat(process.env.MCP_EDIT_MIN_SIMILARITY || '0.7');
        const DEFAULT_CASE_SENSITIVE = process.env.MCP_EDIT_CASE_SENSITIVE === 'true';
        const DEFAULT_IGNORE_WHITESPACE = process.env.MCP_EDIT_IGNORE_WHITESPACE !== 'false';

        return {
            allowedDirectories: args,
            fuzzyMatching: {
                maxDistanceRatio: DEFAULT_MAX_DISTANCE_RATIO,
                minSimilarity: DEFAULT_MIN_SIMILARITY,
                caseSensitive: DEFAULT_CASE_SENSITIVE,
                ignoreWhitespace: DEFAULT_IGNORE_WHITESPACE,
                preserveLeadingWhitespace: 'auto'
            },
            logging: {
                level: (process.env.LOG_LEVEL as any) || 'info',
                performance: process.env.LOG_PERFORMANCE === 'true'
            },
            concurrency: {
                maxConcurrentEdits: parseInt(process.env.MAX_CONCURRENT_EDITS || '10')
            }
        };
    }
}
```
#### Plik: `src/server/index.ts`
```ts
#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import * as pino from 'pino';
import fs from 'node:fs/promises';
import path from 'node:path';

import { loadConfig } from './config.js';
import { setupToolHandlers } from '../core/toolHandlers.js';
import * as schemas from '../core/schemas.js';
import { expandHome, normalizePath } from '../utils/pathUtils.js';

async function main() {
  const config = await loadConfig();

  const logger = pino.pino({
    level: config.logging.level,
    formatters: { level: (label: string) => ({ level: label }) },
    timestamp: () => `,"timestamp":"${new Date().toISOString()}"`,
    base: { service: 'mcp-filesystem-server', version: '0.7.0' }
  });

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
#### Plik: `src/utils/binaryDetect.ts`
```ts
import path from 'node:path';
import { isUtf8 as bufferIsUtf8 } from 'buffer';

const BINARY_EXTENSIONS = new Set([
  '.exe', '.dll', '.so', '.dylib', '.bin', '.dat',
  '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.ico',
  '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
  '.pdf', '.zip', '.rar', '.tar', '.gz', '.7z',
  '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'
]);

export function isBinaryFile(buffer: Buffer, filename?: string): boolean {
  const isUtf8 = (Buffer as any).isUtf8 ?? bufferIsUtf8;
  if (isUtf8 && !isUtf8(buffer)) {
    return true;
  }

  if (buffer.includes(0)) {
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
#### Plik: `tsconfig.json`
```json
{
  "compilerOptions": {
    "target": "es2022",
    "module": "nodenext",
    "moduleResolution": "nodenext", // or "node" if preferred
    "esModuleInterop": true,
    "strict": true,
    "skipLibCheck": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "declaration": true,
    "sourceMap": true,
    "resolveJsonModule": true, // Good for importing config.example.json if needed
    "baseUrl": ".",
    "forceConsistentCasingInFileNames": true
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
---