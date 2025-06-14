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
  allowedDirectories: string[],
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
          await validatePath(fullPath, allowedDirectories, logger, config);

          const relativePath = path.relative(rootPath, fullPath);
          const shouldExclude = excludePatterns.some(p => minimatch(relativePath, p, { dot: true }));

          if (shouldExclude) {
            continue;
          }

          if (entry.name.toLowerCase().includes(pattern.toLowerCase())) {
            results.push(fullPath);
          }

          if (entry.isDirectory()) {
            await search(fullPath);
          }
        } catch (error) {
          logger.debug({ path: fullPath, error: (error as Error).message }, 'Skipping invalid path during search');
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
