import { validatePath } from './security';
import { createError } from '../types/errors';
import path from 'node:path';
import pino from 'pino'; // Import pino
import { Config } from '../server/config'; // Import Config type
import { DEFAULT_EXCLUDE_PATTERNS, DEFAULT_ALLOWED_EXTENSIONS } from '../constants/extensions';

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