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
