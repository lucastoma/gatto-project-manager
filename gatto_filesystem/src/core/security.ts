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
