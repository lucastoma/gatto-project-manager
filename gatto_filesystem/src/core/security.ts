import fs from 'node:fs/promises';
import path from 'node:path';
import { PerformanceTimer } from '../utils/performance';
import { expandHome, normalizePath } from '../utils/pathUtils';
import { createError } from '../types/errors';
import type { Logger } from 'pino';
import type { Config } from '../server/config';

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

