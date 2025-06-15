import { shouldSkipPath } from './pathFilter.js';
import { DEFAULT_EXCLUDE_PATTERNS, DEFAULT_ALLOWED_EXTENSIONS } from '../constants/extensions.js';

import { Config } from '../server/config.js'; // Import the Config type

// Define the type for overrides to allow partial nested objects
type MockConfigOverrides = {
  allowedDirectories?: string[];
  fileFiltering?: Partial<Config['fileFiltering']>;
  fuzzyMatching?: Partial<Config['fuzzyMatching']>;
  logging?: Partial<Config['logging']>;
  concurrency?: Partial<Config['concurrency']>;
  limits?: Partial<Config['limits']>;
};

describe('shouldSkipPath', () => {
  const mockConfig = (overrides: MockConfigOverrides = {}): Config => ({
    allowedDirectories: overrides.allowedDirectories || ['/allowed'],
    fileFiltering: {
      defaultExcludes: overrides.fileFiltering?.defaultExcludes ?? DEFAULT_EXCLUDE_PATTERNS,
      allowedExtensions: overrides.fileFiltering?.allowedExtensions ?? DEFAULT_ALLOWED_EXTENSIONS,
      forceTextFiles: overrides.fileFiltering?.forceTextFiles ?? true, // Default from ConfigSchema (true)
    },
    fuzzyMatching: {
      maxDistanceRatio: overrides.fuzzyMatching?.maxDistanceRatio ?? 0.25,
      minSimilarity: overrides.fuzzyMatching?.minSimilarity ?? 0.7,
      caseSensitive: overrides.fuzzyMatching?.caseSensitive ?? false,
      ignoreWhitespace: overrides.fuzzyMatching?.ignoreWhitespace ?? true,
      preserveLeadingWhitespace: (overrides.fuzzyMatching?.preserveLeadingWhitespace ?? 'auto') as 'auto' | 'strict' | 'normalize', // Matches ConfigSchema
    },
    logging: {
      level: (overrides.logging?.level ?? 'info') as 'trace' | 'debug' | 'info' | 'warn' | 'error', // Matches ConfigSchema
      performance: overrides.logging?.performance ?? false,
    },
    concurrency: {
      maxConcurrentEdits: overrides.concurrency?.maxConcurrentEdits ?? 10,
      maxGlobalConcurrentEdits: overrides.concurrency?.maxGlobalConcurrentEdits ?? 20,
    },
    limits: {
      maxReadBytes: overrides.limits?.maxReadBytes ?? 5 * 1024 * 1024,
      maxWriteBytes: overrides.limits?.maxWriteBytes ?? 5 * 1024 * 1024,
    }
  });


  it('should allow basic file when no filters', () => {
    expect(shouldSkipPath('/allowed/file.txt', mockConfig())).toBe(false);
  });

  it('should exclude files matching default patterns', () => {
    DEFAULT_EXCLUDE_PATTERNS.forEach(pattern => {
      const testPath = `/allowed/${pattern.replace('*', 'test')}`;
      expect(shouldSkipPath(testPath, mockConfig())).toBe(true);
    });
  });

  it('should exclude files matching custom patterns', () => {
    const config = mockConfig({ fileFiltering: { defaultExcludes: ['custom*'] } });
    expect(shouldSkipPath('/allowed/custom-file.txt', config)).toBe(true);
    expect(shouldSkipPath('/allowed/other-file.txt', config)).toBe(false);
  });

  it('should allow files with default extensions when forceTextFiles=true', () => {
    const config = mockConfig({ fileFiltering: { forceTextFiles: true } });
    DEFAULT_ALLOWED_EXTENSIONS.forEach(ext => {
      expect(shouldSkipPath(`/allowed/file${ext}`, config)).toBe(false);
    });
  });

  it('should exclude files without allowed extensions when forceTextFiles=true', () => {
    const config = mockConfig({ fileFiltering: { forceTextFiles: true } });
    expect(shouldSkipPath('/allowed/file.bin', config)).toBe(true);
  });

  it('should allow files with custom extensions when forceTextFiles=true', () => {
    const config = mockConfig({
      fileFiltering: {
        forceTextFiles: true,
        allowedExtensions: ['*.custom']
      }
    });
    expect(shouldSkipPath('/allowed/file.custom', config)).toBe(false);
    expect(shouldSkipPath('/allowed/file.txt', config)).toBe(true);
  });

  it('should handle case insensitivity', () => {
    const config = mockConfig({ fileFiltering: { defaultExcludes: ['UPPER*'] } });
    expect(shouldSkipPath('/allowed/upper-case.txt', config)).toBe(true);
    expect(shouldSkipPath('/allowed/UPPER-CASE.txt', config)).toBe(true);
  });

  it('should handle dot files', () => {
    const config = mockConfig({ fileFiltering: { defaultExcludes: ['.*'] } });
    expect(shouldSkipPath('/allowed/.hidden', config)).toBe(true);
    expect(shouldSkipPath('/allowed/visible', config)).toBe(false);
  });
});