import { validatePath } from './security.js';
import { createError } from '../types/errors.js';
import path from 'node:path';

describe('validatePath', () => {
  const mockLogger = {
    info: jest.fn(),
    debug: jest.fn(),
    error: jest.fn()
  };
  
  const mockConfig = {
    allowedDirectories: ['/allowed'],
    fileFiltering: {
      defaultExcludes: [],
      allowedExtensions: [],
      forceTextFiles: false
    },
    logging: { level: 'info' }
  };

  it('should allow paths within allowed directories', async () => {
    const result = await validatePath('/allowed/file.txt', ['/allowed'], mockLogger, mockConfig);
    expect(result).toBe(path.normalize('/allowed/file.txt'));
  });

  it('should reject paths outside allowed directories', async () => {
    await expect(validatePath('/outside/file.txt', ['/allowed'], mockLogger, mockConfig))
      .rejects
      .toThrow(createError('ACCESS_DENIED', expect.any(String)));
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