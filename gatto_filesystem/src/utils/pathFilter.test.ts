import { shouldSkipPath } from './pathFilter.js';
import { DEFAULT_EXCLUDE_PATTERNS, DEFAULT_ALLOWED_EXTENSIONS } from '../constants/extensions.js';

describe('shouldSkipPath', () => {
  const mockConfig = (overrides = {}) => ({
    allowedDirectories: ['/allowed'],
    fileFiltering: {
      defaultExcludes: [],
      allowedExtensions: [],
      forceTextFiles: false,
      ...overrides
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