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
    const absolute = path.isAbsolute(expandedPath)
      ? path.resolve(expandedPath)
      : path.resolve(process.cwd(), expandedPath);

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
      const parentDir = path.dirname(absolute);
      try {
        const realParentPath = await fs.realpath(parentDir);
        const normalizedParent = normalizePath(realParentPath);
        const isParentAllowed = allowedDirectories.some(dir => {
          const relativePath = path.relative(dir, normalizedParent);
          return !relativePath.startsWith('..' + path.sep) && relativePath !== '..';
        });
        
        if (!isParentAllowed) {
          throw createError(
            'ACCESS_DENIED',
            'Parent directory outside allowed directories',
            { parentDirectory: realParentPath }
          );
        }
        
        timer.end({ result: 'success', newFile: true });
        return absolute;
      } catch {
        throw createError(
          'ACCESS_DENIED',
          `Parent directory does not exist or is not accessible: ${parentDir}`,
          { parentDirectory: parentDir }
        );
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
