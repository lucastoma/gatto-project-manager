import fs from 'node:fs/promises';
import path from 'node:path';
import { minimatch } from 'minimatch';
import { lookup as mimeLookup } from 'mime-types';
import { classifyFileType, FileType } from '../utils/binaryDetect.js';
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
        const fileType = classifyFileType(actualBuffer, filePath);
        isBinary = fileType === FileType.CONFIRMED_BINARY;
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
        const fileType = classifyFileType(checkBuffer, validPath);
        if (fileType === FileType.CONFIRMED_BINARY || fileType === FileType.POTENTIAL_TEXT_WITH_CAVEATS) {
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
