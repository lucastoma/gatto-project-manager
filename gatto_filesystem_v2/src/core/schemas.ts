import { z } from 'zod';
import type { Config } from '../server/config.js';

export const ReadFileArgsSchema = z.object({
  path: z.string(),
  encoding: z.enum(['utf-8', 'base64', 'auto']).default('auto').describe('Encoding for file content')
});

export const ReadMultipleFilesArgsSchema = z.object({
  paths: z.array(z.string()),
  encoding: z.enum(['utf-8', 'base64', 'auto']).default('auto').describe('Encoding for file content')
});

export const WriteFileArgsSchema = z.object({
  path: z.string(),
  content: z.string(),
  encoding: z.enum(['utf-8', 'base64']).default('utf-8').describe('Encoding of provided content')
});

const EditOperation = z.object({
  oldText: z.string().describe('Text to search for - can be slightly inaccurate'),
  newText: z.string().describe('Text to replace with')
});

export const getEditFileArgsSchema = (config: Config) => z.object({
  path: z.string(),
  edits: z.array(EditOperation),
  dryRun: z.boolean().default(false).describe('Preview changes using git-style diff format'),
  debug: z.boolean().default(false).describe('Show detailed matching information'),
  caseSensitive: z.boolean().default(config.fuzzyMatching.caseSensitive).describe('Whether to match case sensitively'),
  ignoreWhitespace: z.boolean().default(config.fuzzyMatching.ignoreWhitespace).describe('Whether to normalize whitespace differences'),
  maxDistanceRatio: z.number().min(0).max(1).default(config.fuzzyMatching.maxDistanceRatio).describe('Maximum allowed distance as ratio of text length'),
  minSimilarity: z.number().min(0).max(1).default(config.fuzzyMatching.minSimilarity).describe('Minimum similarity threshold (0-1)'),
  preserveLeadingWhitespace: z.enum(['auto', 'strict', 'normalize']).default(config.fuzzyMatching.preserveLeadingWhitespace).describe('How to handle leading whitespace preservation')
});

export const CreateDirectoryArgsSchema = z.object({
  path: z.string(),
});

export const ListDirectoryArgsSchema = z.object({
  path: z.string(),
});

export const DirectoryTreeArgsSchema = z.object({
  path: z.string(),
});

export const MoveFileArgsSchema = z.object({
  source: z.string(),
  destination: z.string(),
});

export const SearchFilesArgsSchema = z.object({
  path: z.string(),
  pattern: z.string(),
  excludePatterns: z.array(z.string()).optional().default([]),
  useExactPatterns: z.boolean().default(false).describe('Use patterns exactly as provided instead of wrapping with **/')
});

export const GetFileInfoArgsSchema = z.object({
  path: z.string(),
});
