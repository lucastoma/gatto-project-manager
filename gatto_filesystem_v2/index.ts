#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ToolSchema,
} from "@modelcontextprotocol/sdk/types.js";
import fs from "node:fs/promises";
import path from "node:path";
import os from 'node:os';
import { performance } from 'node:perf_hooks';
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import { createTwoFilesPatch } from 'diff';
import { minimatch } from 'minimatch';
import pino from 'pino';
import { Mutex } from 'async-mutex';

// Error types for structured error responses
interface StructuredError {
  code: string;
  message: string;
  details?: any;
}

// Configuration schema
const ConfigSchema = z.object({
  allowedDirectories: z.array(z.string()),
  fuzzyMatching: z.object({
    maxDistanceRatio: z.number().min(0).max(1).default(0.25),
    minSimilarity: z.number().min(0).max(1).default(0.7),
    caseSensitive: z.boolean().default(false),
    ignoreWhitespace: z.boolean().default(true),
    preserveLeadingWhitespace: z.enum(['auto', 'strict', 'normalize']).default('auto')
  }).default({}),
  logging: z.object({
    level: z.enum(['trace', 'debug', 'info', 'warn', 'error']).default('info'),
    performance: z.boolean().default(false)
  }).default({}),
  concurrency: z.object({
    maxConcurrentEdits: z.number().positive().default(10)
  }).default({})
});

type Config = z.infer<typeof ConfigSchema>;

// Command line argument parsing with config file support
const args = process.argv.slice(2);
let config: Config;

if (args.length > 0 && (args[0] === '--config' || args[0] === '-c')) {
  if (args.length < 2) {
    console.error("Usage: mcp-server-filesystem --config <config-file>");
    console.error("   or: mcp-server-filesystem <allowed-directory> [additional-directories...]");
    process.exit(1);
  }
  
  try {
    const configPath = path.resolve(args[1]);
    const configContent = await fs.readFile(configPath, 'utf-8');
    const rawConfig = JSON.parse(configContent);
    config = ConfigSchema.parse(rawConfig);
  } catch (error) {
    console.error("Error loading config file:", error);
    process.exit(1);
  }
} else {
  if (args.length === 0) {
    console.error("Usage: mcp-server-filesystem --config <config-file>");
    console.error("   or: mcp-server-filesystem <allowed-directory> [additional-directories...]");
    process.exit(1);
  }
  
  // Fallback to CLI arguments
  const DEFAULT_MAX_DISTANCE_RATIO = parseFloat(process.env.MCP_EDIT_MAX_DISTANCE_RATIO || '0.25');
  const DEFAULT_MIN_SIMILARITY = parseFloat(process.env.MCP_EDIT_MIN_SIMILARITY || '0.7');
  const DEFAULT_CASE_SENSITIVE = process.env.MCP_EDIT_CASE_SENSITIVE === 'true';
  const DEFAULT_IGNORE_WHITESPACE = process.env.MCP_EDIT_IGNORE_WHITESPACE !== 'false';

  config = {
    allowedDirectories: args,
    fuzzyMatching: {
      maxDistanceRatio: DEFAULT_MAX_DISTANCE_RATIO,
      minSimilarity: DEFAULT_MIN_SIMILARITY,
      caseSensitive: DEFAULT_CASE_SENSITIVE,
      ignoreWhitespace: DEFAULT_IGNORE_WHITESPACE,
      preserveLeadingWhitespace: 'auto'
    },
    logging: {
      level: (process.env.LOG_LEVEL as any) || 'info',
      performance: process.env.LOG_PERFORMANCE === 'true'
    },
    concurrency: {
      maxConcurrentEdits: parseInt(process.env.MAX_CONCURRENT_EDITS || '10')
    }
  };
}

// Logger setup
const logger = pino({
  level: config.logging.level,
  formatters: {
    level: (label) => ({ level: label }),
  },
  timestamp: () => `,"timestamp":"${new Date().toISOString()}"`,
  base: {
    service: 'mcp-filesystem-server',
    version: '0.5.0'
  }
});

// Performance timing utility
class PerformanceTimer {
  private startTime: number;
  private operation: string;

  constructor(operation: string) {
    this.operation = operation;
    this.startTime = performance.now();
  }

  end(additionalData?: any): number {
    const duration = performance.now() - this.startTime;
    if (config.logging.performance) {
      logger.debug({
        operation: this.operation,
        duration_ms: Math.round(duration * 100) / 100,
        ...additionalData
      }, `Performance: ${this.operation}`);
    }
    return duration;
  }
}

// File operation concurrency control
const fileLocks = new Map<string, Mutex>();
const maxConcurrentLocks = config.concurrency.maxConcurrentEdits;

function getFileLock(filePath: string): Mutex {
  if (!fileLocks.has(filePath)) {
    if (fileLocks.size >= maxConcurrentLocks) {
      // Clean up unused locks (simple LRU-like behavior)
      const oldestKey = fileLocks.keys().next().value;
      fileLocks.delete(oldestKey);
    }
    fileLocks.set(filePath, new Mutex());
  }
  return fileLocks.get(filePath)!;
}

// Server statistics
let requestCount = 0;
let editOperationCount = 0;
let binaryFileAttempts = 0;
let averageEditTime = 0;

// Normalize all paths consistently
function normalizePath(p: string): string {
  return path.normalize(p);
}

function expandHome(filepath: string): string {
  if (filepath.startsWith('~/') || filepath === '~') {
    return path.join(os.homedir(), filepath.slice(1));
  }
  return filepath;
}

// Store allowed directories in normalized form
const allowedDirectories = config.allowedDirectories.map(dir =>
  normalizePath(path.resolve(expandHome(dir)))
);

// Validate that all directories exist and are accessible
await Promise.all(config.allowedDirectories.map(async (dir) => {
  try {
    const stats = await fs.stat(expandHome(dir));
    if (!stats.isDirectory()) {
      logger.error(`Error: ${dir} is not a directory`);
      process.exit(1);
    }
  } catch (error) {
    logger.error({ error, directory: dir }, `Error accessing directory ${dir}`);
    process.exit(1);
  }
}));

// Binary file detection utilities
function isBinaryFile(buffer: Buffer, filename?: string): boolean {
  // Check for UTF-8 validity first (Node.js 18+ feature)
  if (Buffer.isUtf8 && !Buffer.isUtf8(buffer)) {
    return true;
  }

  // Check for null bytes (common in binary files)
  if (buffer.includes(0)) {
    return true;
  }

  // Check file extension
  if (filename) {
    const ext = path.extname(filename).toLowerCase();
    const binaryExtensions = new Set([
      '.exe', '.dll', '.so', '.dylib', '.bin', '.dat',
      '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.ico',
      '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
      '.pdf', '.zip', '.rar', '.tar', '.gz', '.7z',
      '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'
    ]);
    
    if (binaryExtensions.has(ext)) {
      return true;
    }
  }

  // Check for high ratio of non-printable characters
  let nonPrintable = 0;
  const sampleSize = Math.min(1024, buffer.length);
  
  for (let i = 0; i < sampleSize; i++) {
    const byte = buffer[i];
    if (byte < 32 && byte !== 9 && byte !== 10 && byte !== 13) {
      nonPrintable++;
    }
  }

  return (nonPrintable / sampleSize) > 0.1; // More than 10% non-printable
}

// Create structured error
function createError(code: string, message: string, details?: any): StructuredError {
  return { code, message, details };
}

// Security utilities with improved path validation
async function validatePath(requestedPath: string): Promise<string> {
  const timer = new PerformanceTimer('validatePath');
  
  try {
    const expandedPath = expandHome(requestedPath);
    const absolute = path.isAbsolute(expandedPath)
      ? path.resolve(expandedPath)
      : path.resolve(process.cwd(), expandedPath);

    const normalizedRequested = normalizePath(absolute);

    // Check if path is within allowed directories using safer method
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

    // Handle symlinks by checking their real path
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
      // For new files that don't exist yet, verify parent directory
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
          'PARENT_NOT_FOUND',
          `Parent directory does not exist: ${parentDir}`,
          { parentDirectory: parentDir }
        );
      }
    }
  } catch (error) {
    timer.end({ result: 'error' });
    if ((error as any).code) {
      throw error; // Already a structured error
    }
    throw createError('VALIDATION_ERROR', (error as Error).message || String(error));
  }
}

// Schema definitions
const ReadFileArgsSchema = z.object({
  path: z.string(),
  encoding: z.enum(['utf-8', 'base64', 'auto']).default('auto').describe('Encoding for file content')
});

const ReadMultipleFilesArgsSchema = z.object({
  paths: z.array(z.string()),
  encoding: z.enum(['utf-8', 'base64', 'auto']).default('auto').describe('Encoding for file content')
});

const WriteFileArgsSchema = z.object({
  path: z.string(),
  content: z.string(),
  encoding: z.enum(['utf-8', 'base64']).default('utf-8').describe('Encoding of provided content')
});

const EditOperation = z.object({
  oldText: z.string().describe('Text to search for - can be slightly inaccurate'),
  newText: z.string().describe('Text to replace with')
});

const EditFileArgsSchema = z.object({
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

const CreateDirectoryArgsSchema = z.object({
  path: z.string(),
});

const ListDirectoryArgsSchema = z.object({
  path: z.string(),
});

const DirectoryTreeArgsSchema = z.object({
  path: z.string(),
});

const MoveFileArgsSchema = z.object({
  source: z.string(),
  destination: z.string(),
});

const SearchFilesArgsSchema = z.object({
  path: z.string(),
  pattern: z.string(),
  excludePatterns: z.array(z.string()).optional().default([]),
  useExactPatterns: z.boolean().default(false).describe('Use patterns exactly as provided instead of wrapping with **/')
});

const GetFileInfoArgsSchema = z.object({
  path: z.string(),
});

const ToolInputSchema = ToolSchema.shape.inputSchema;
type ToolInput = z.infer<typeof ToolInputSchema>;

interface FileInfo {
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

interface FuzzyMatchConfig {
  maxDistanceRatio: number;
  minSimilarity: number;
  caseSensitive: boolean;
  ignoreWhitespace: boolean;
  preserveLeadingWhitespace: 'auto' | 'strict' | 'normalize';
  debug: boolean;
}

interface ApplyFileEditsResult {
  modifiedContent: string;
  formattedDiff: string;
}

// Server setup
const server = new Server(
  {
    name: "secure-filesystem-server",
    version: "0.5.0",
  },
  {
    capabilities: {
      tools: {},
    },
  },
);

// Tool implementations
async function getFileStats(filePath: string): Promise<FileInfo> {
  const timer = new PerformanceTimer('getFileStats');
  
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

// Symlink loop detection using visited inodes
async function searchFiles(
  rootPath: string,
  pattern: string,
  excludePatterns: string[] = [],
  useExactPatterns: boolean = false
): Promise<string[]> {
  const timer = new PerformanceTimer('searchFiles');
  const results: string[] = [];
  const visitedForThisSearch = new Set<string>();

  async function search(currentPath: string): Promise<void> {
    try {
      const stats = await fs.stat(currentPath);
      const inodeKey = `${stats.dev}-${stats.ino}`;
      
      if (visitedForThisSearch.has(inodeKey)) {
        logger.debug({ path: currentPath }, 'Skipping already visited inode (symlink loop)');
        return; // Avoid infinite loops with symlinks
      }
      visitedForThisSearch.add(inodeKey);

      const entries = await fs.readdir(currentPath, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(currentPath, entry.name);

        try {
          // Validate each path before processing
          await validatePath(fullPath);

          // Check if path matches any exclude pattern
          const relativePath = path.relative(rootPath, fullPath);
          const shouldExclude = excludePatterns.some(pattern => {
            const globPattern = useExactPatterns ? pattern : (pattern.includes('*') ? pattern : `**/${pattern}/**`);
            return minimatch(relativePath, globPattern, { dot: true, matchBase: !useExactPatterns });
          });

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
          // Skip invalid paths during search
          logger.debug({ path: fullPath, error: (error as Error).message }, 'Skipping invalid path during search');
          continue;
        }
      }
    } catch (error) {
      // Skip paths that can't be accessed
      logger.debug({ path: currentPath, error: (error as Error).message }, 'Skipping inaccessible path');
      return;
    }
  }

  await search(rootPath);
  timer.end({ resultsCount: results.length });
  return results;
}

// file editing and diffing utilities
function normalizeLineEndings(text: string): string {
  // Handle all three line ending variants: \r\n, \r, \n
  return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
}

function createUnifiedDiff(originalContent: string, newContent: string, filepath: string = 'file'): string {
  // Ensure consistent line endings for diff
  const normalizedOriginal = normalizeLineEndings(originalContent);
  const normalizedNew = normalizeLineEndings(newContent);

  return createTwoFilesPatch(
    filepath,
    filepath,
    normalizedOriginal,
    normalizedNew,
    'original',
    'modified'
  );
}

/**
 * Preprocesses text for fuzzy matching based on configuration
 */
function preprocessText(text: string, config: FuzzyMatchConfig): string {
  let processed = text;
  
  if (!config.caseSensitive) {
    processed = processed.toLowerCase();
  }
  
  if (config.ignoreWhitespace) {
    // Normalize whitespace but preserve structure
    processed = processed.replace(/[ \t]+/g, ' ').replace(/\n+/g, '\n').trim();
  }
  
  return processed;
}

/**
 * Optimized Levenshtein distance calculation with memory optimization for long texts
 */
function levenshteinDistanceOptimized(str1: string, str2: string): number {
  if (str1 === str2) return 0;
  if (str1.length === 0) return str2.length;
  if (str2.length === 0) return str1.length;
  
  // Swap strings so that str1 is always the shorter one for memory optimization
  const shorter = str1.length <= str2.length ? str1 : str2;
  const longer = str1.length <= str2.length ? str2 : str1;
  
  // Use only two rows instead of full matrix to save memory
  let previousRow = Array(shorter.length + 1).fill(0).map((_, i) => i);
  
  for (let i = 0; i < longer.length; i++) {
    const currentRow = [i + 1];
    
    for (let j = 0; j < shorter.length; j++) {
      const cost = longer[i] === shorter[j] ? 0 : 1;
      currentRow.push(Math.min(
        currentRow[j] + 1,           // insertion
        previousRow[j + 1] + 1,      // deletion
        previousRow[j] + cost        // substitution
      ));
    }
    
    previousRow = currentRow;
  }
  
  return previousRow[shorter.length];
}

/**
 * Calculates similarity score from Levenshtein distance
 */
function calculateSimilarity(distance: number, maxLength: number): number {
  return Math.max(0, 1 - (distance / maxLength));
}

/**
 * Enhanced validation that edits don't have problematic overlaps
 */
function validateEdits(edits: Array<{oldText: string, newText: string}>, debug: boolean): void {
  for (let i = 0; i < edits.length; i++) {
    for (let j = i + 1; j < edits.length; j++) {
      const edit1 = edits[i];
      const edit2 = edits[j];
      
      // Check for oldText overlaps
      if (edit1.oldText.includes(edit2.oldText) || edit2.oldText.includes(edit1.oldText)) {
        const warning = `Warning: Potentially overlapping oldText in edits ${i+1} and ${j+1}`;
        logger.warn({ edit1Index: i, edit2Index: j }, warning);
        if (debug) {
          console.error(warning);
        }
      }
      
      // Check for newText conflicts that might create mutual overlaps
      if (edit1.newText.includes(edit2.oldText) || edit2.newText.includes(edit1.oldText)) {
        const warning = `Warning: newText in edit ${i+1} contains oldText from edit ${j+1} - potential mutual overlap`;
        logger.warn({ edit1Index: i, edit2Index: j }, warning);
        if (debug) {
          console.error(warning);
        }
      }
    }
  }
}

/**
 * Applies relative indentation based on the original block structure and configuration
 */
function applyRelativeIndentation(
  newLines: string[], 
  oldLines: string[], 
  originalIndent: string,
  preserveMode: 'auto' | 'strict' | 'normalize'
): string[] {
  switch (preserveMode) {
    case 'strict':
      // Keep original indentation exactly as is
      return newLines.map(line => originalIndent + line.trimStart());
      
    case 'normalize':
      // Apply consistent indentation based on the first line
      return newLines.map(line => originalIndent + line.trimStart());
      
    case 'auto':
    default:
      // Smart indentation preservation (original behavior)
      return newLines.map((line, idx) => {
        if (idx === 0) {
          return originalIndent + line.trimStart();
        }
        
        // For subsequent lines, maintain relative indentation
        const oldLineIndex = Math.min(idx, oldLines.length - 1);
        const newLineIndent = line.match(/^\s*/)?.[0] || '';
        
        // Calculate relative indentation change
        const baseOldIndent = oldLines[0]?.match(/^\s*/)?.[0]?.length || 0;
        const relativeIndentChange = newLineIndent.length - baseOldIndent;
        const finalIndent = originalIndent + ' '.repeat(Math.max(0, relativeIndentChange));
        
        return finalIndent + line.trimStart();
      });
  }
}

async function applyFileEdits(
  filePath: string,
  edits: Array<{oldText: string, newText: string}>,
  config: FuzzyMatchConfig
): Promise<ApplyFileEditsResult> {
  const timer = new PerformanceTimer('applyFileEdits');
  let levenshteinIterations = 0;
  
  try {
    // Check if file is binary first
    const buffer = await fs.readFile(filePath);
    if (isBinaryFile(buffer, filePath)) {
      binaryFileAttempts++;
      throw createError(
        'BINARY_FILE_ERROR',
        'Cannot edit binary files',
        { filePath, detectedAs: 'binary' }
      );
    }

    const originalContent = normalizeLineEndings(buffer.toString('utf-8'));
    let modifiedContent = originalContent;

    // Validate edits for potential issues
    validateEdits(edits, config.debug);

    for (const [editIndex, edit] of edits.entries()) {
      const normalizedOld = normalizeLineEndings(edit.oldText);
      const normalizedNew = normalizeLineEndings(edit.newText);
      let matchFound = false;

      if (config.debug) {
        logger.debug({ editIndex, totalEdits: edits.length, oldTextPreview: normalizedOld.substring(0, 50) }, 'Processing edit');
      }

      // 1. Try for an exact match first for performance and precision
      const exactMatchIndex = modifiedContent.indexOf(normalizedOld);
      if (exactMatchIndex !== -1) {
        modifiedContent = modifiedContent.substring(0, exactMatchIndex) +
                          normalizedNew +
                          modifiedContent.substring(exactMatchIndex + normalizedOld.length);
        matchFound = true;
        
        if (config.debug) {
          logger.debug({ exactMatchIndex }, 'Exact match found');
        }
      } else {
        // 2. If no exact match, use fuzzy matching with optimized Levenshtein distance
        const contentLines = modifiedContent.split('\n');
        const oldLines = normalizedOld.split('\n');

        // Preprocess text for comparison
        const processedOld = preprocessText(normalizedOld, config);

        let bestMatch = {
          distance: Infinity,
          index: -1,
          text: '',
          similarity: 0,
          windowSize: 0
        };

        // Try different window sizes to handle cases where line count differs
        const windowSizes = [
          oldLines.length,
          Math.max(1, oldLines.length - 1),
          oldLines.length + 1,
          Math.max(1, oldLines.length - 2),
          oldLines.length + 2
        ].filter((size, index, arr) => arr.indexOf(size) === index && size > 0); // Remove duplicates and invalid sizes

        for (const windowSize of windowSizes) {
          if (windowSize > contentLines.length) continue;

          // Slide a window over the content to find the best fuzzy match
          for (let i = 0; i <= contentLines.length - windowSize; i++) {
            const windowLines = contentLines.slice(i, i + windowSize);
            const windowText = windowLines.join('\n');
            const processedWindow = preprocessText(windowText, config);
            
            levenshteinIterations++;
            const distance = levenshteinDistanceOptimized(processedOld, processedWindow);
            const similarity = calculateSimilarity(distance, Math.max(processedOld.length, processedWindow.length));

            if (distance < bestMatch.distance) {
              bestMatch = { 
                distance, 
                index: i, 
                text: windowText,
                similarity,
                windowSize
              };
            }
            
            if (distance === 0) break; // Perfect match found
          }
          
          if (bestMatch.distance === 0) break; // Perfect match found, no need to try other window sizes
        }

        // Check if the best match meets our similarity threshold
        const distanceThreshold = Math.floor(processedOld.length * config.maxDistanceRatio);

        if (config.debug) {
          logger.debug({
            bestMatchDistance: bestMatch.distance,
            bestMatchSimilarity: bestMatch.similarity,
            similarityThreshold: config.minSimilarity,
            distanceThreshold,
            windowSize: bestMatch.windowSize,
            levenshteinIterations
          }, 'Fuzzy matching results');
        }

        if (bestMatch.index !== -1 && 
            bestMatch.distance <= distanceThreshold && 
            bestMatch.similarity >= config.minSimilarity) {
          
          const newLines = normalizedNew.split('\n');
          
          // Preserve and apply relative indentation from the original file's matched block
          const originalIndent = contentLines[bestMatch.index].match(/^\s*/)?.[0] || '';
          const indentedNewLines = applyRelativeIndentation(
            newLines, 
            oldLines, 
            originalIndent, 
            config.preserveLeadingWhitespace
          );

          // Replace the matched lines with the new content
          contentLines.splice(bestMatch.index, bestMatch.windowSize, ...indentedNewLines);
          modifiedContent = contentLines.join('\n');
          matchFound = true;

          if (config.debug) {
            logger.debug({ similarity: bestMatch.similarity }, 'Fuzzy match applied successfully');
          }
        }
      }

      if (!matchFound) {
        let errorMessage = `Could not find a close match for edit ${editIndex + 1}:\n---\n${edit.oldText}\n---`;
        
        // Try to provide helpful suggestions
        const searchTerm = edit.oldText.trim().split('\n')[0].substring(0, 50);
        if (searchTerm) {
          const suggestions = [];
          const lines = modifiedContent.split('\n');
          
          for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            if (line.toLowerCase().includes(searchTerm.toLowerCase())) {
              const start = Math.max(0, i - 1);
              const end = Math.min(lines.length, i + 2);
              suggestions.push(`Line ${i + 1}: ${lines.slice(start, end).join('\n')}`);
              if (suggestions.length >= 3) break; // Limit suggestions
            }
          }
          
          if (suggestions.length > 0) {
            errorMessage += `\n\nDid you mean something like:\n---\n${suggestions.join('\n---\n')}\n---`;
          }
        }
        
        throw createError(
          'FUZZY_MATCH_FAILED',
          errorMessage,
          { editIndex, searchTerm, levenshteinIterations }
        );
      }
    }

    // Create unified diff
    const diff = createUnifiedDiff(originalContent, modifiedContent, filePath);

    // Format diff with appropriate number of backticks
    let numBackticks = 3;
    while (diff.includes('`'.repeat(numBackticks))) {
      numBackticks++;
    }
    const formattedDiff = `${'`'.repeat(numBackticks)}diff\n${diff}\n${'`'.repeat(numBackticks)}\n\n`;

    const duration = timer.end({ 
      editsCount: edits.length, 
      levenshteinIterations,
      charactersProcessed: originalContent.length
    });

    // Update average edit time
    averageEditTime = (averageEditTime * editOperationCount + duration) / (editOperationCount + 1);

    return { modifiedContent, formattedDiff };
  } catch (error) {
    timer.end({ result: 'error' });
    throw error;
  }
}

// Tool handlers... (reszta kodu jest identyczna, tylko dodane type assertions gdzie potrzebne)

// Tool handlers
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "read_file",
        description:
          "Read the complete contents of a file from the file system. " +
          "Automatically detects binary files and serves them as Base64 to prevent terminal corruption. " +
          "Handles various text encodings and provides detailed error messages " +
          "if the file cannot be read. Use this tool when you need to examine " +
          "the contents of a single file. Only works within allowed directories.",
        inputSchema: zodToJsonSchema(ReadFileArgsSchema) as ToolInput,
      },
      {
        name: "read_multiple_files",
        description:
          "Read the contents of multiple files simultaneously with automatic binary detection. " +
          "This is more efficient than reading files one by one when you need to analyze " +
          "or compare multiple files. Each file's content is returned with its " +
          "path as a reference. Binary files are automatically encoded as Base64. " +
          "Failed reads for individual files won't stop the entire operation. Only works within allowed directories.",
        inputSchema: zodToJsonSchema(ReadMultipleFilesArgsSchema) as ToolInput,
      },
      {
        name: "write_file",
        description:
          "Create a new file or completely overwrite an existing file with new content. " +
          "Use with caution as it will overwrite existing files without warning. " +
          "Supports both UTF-8 text and Base64 encoded content for binary files. " +
          "Handles text content with proper encoding. Only works within allowed directories.",
        inputSchema: zodToJsonSchema(WriteFileArgsSchema) as ToolInput,
      },
      {
        name: "edit_file",
        description:
          "Makes specific, line-based edits to text files using advanced fuzzy matching with concurrency control. " +
          "Features: automatic binary file detection and rejection, exact match priority, " +
          "Levenshtein distance-based fuzzy matching, configurable similarity thresholds, " +
          "intelligent indentation preservation, multiple window size attempts, " +
          "file-level locking for concurrent safety, performance monitoring, " +
          "and helpful error suggestions with structured error responses. " +
          "Supports multiple whitespace handling modes and global configuration. " +
          "Returns a git-style diff of the changes. Ideal for robust, fault-tolerant editing " +
          "that handles minor typos and formatting differences. Only works within allowed directories " +
          "and only on text files.",
        inputSchema: zodToJsonSchema(EditFileArgsSchema) as ToolInput,
      },
      {
        name: "create_directory",
        description:
          "Create a new directory or ensure a directory exists. Can create multiple " +
          "nested directories in one operation. If the directory already exists, " +
          "this operation will succeed silently. Perfect for setting up directory " +
          "structures for projects or ensuring required paths exist. Only works within allowed directories.",
        inputSchema: zodToJsonSchema(CreateDirectoryArgsSchema) as ToolInput,
      },
      {
        name: "list_directory",
        description:
          "Get a detailed listing of all files and directories in a specified path. " +
          "Results clearly distinguish between files and directories with [FILE] and [DIR] " +
          "prefixes. This tool is essential for understanding directory structure and " +
          "finding specific files within a directory. Only works within allowed directories.",
        inputSchema: zodToJsonSchema(ListDirectoryArgsSchema) as ToolInput,
      },
      {
        name: "directory_tree",
        description:
            "Get a recursive tree view of files and directories as a JSON structure. " +
            "Each entry includes 'name', 'type' (file/directory), and 'children' for directories. " +
            "Files have no children array, while directories always have a children array (which may be empty). " +
            "Protected against symlink loops using inode tracking with performance monitoring. " +
            "The output is formatted with 2-space indentation for readability. Only works within allowed directories.",
        inputSchema: zodToJsonSchema(DirectoryTreeArgsSchema) as ToolInput,
      },
      {
        name: "move_file",
        description:
          "Move or rename files and directories. Can move files between directories " +
          "and rename them in a single operation. If the destination exists, the " +
          "operation will fail. Works across different directories and can be used " +
          "for simple renaming within the same directory. Both source and destination must be within allowed directories.",
        inputSchema: zodToJsonSchema(MoveFileArgsSchema) as ToolInput,
      },
      {
        name: "search_files",
        description:
          "Recursively search for files and directories matching a pattern with symlink loop protection. " +
          "Searches through all subdirectories from the starting path. The search " +
          "is case-insensitive and matches partial names. Supports both automatic glob wrapping " +
          "and exact pattern usage. Protected against symlink loops with performance monitoring. " +
          "Returns full paths to all matching items. Great for finding files when you don't know their exact location. " +
          "Only searches within allowed directories.",
        inputSchema: zodToJsonSchema(SearchFilesArgsSchema) as ToolInput,
      },
      {
        name: "get_file_info",
        description:
          "Retrieve detailed metadata about a file or directory including binary detection. " +
          "Returns comprehensive information including size, creation time, last modified time, " +
          "permissions, type, binary status, and MIME type detection. " +
          "This tool is perfect for understanding file characteristics " +
          "without reading the actual content. Only works within allowed directories.",
        inputSchema: zodToJsonSchema(GetFileInfoArgsSchema) as ToolInput,
      },
      {
        name: "list_allowed_directories",
        description:
          "Returns the list of directories that this server is allowed to access. " +
          "Use this to understand which directories are available before trying to access files.",
        inputSchema: {
          type: "object",
          properties: {},
          required: [],
        },
      },
      {
        name: "server_stats",
        description:
          "Returns comprehensive server statistics and current configuration including request counts, " +
          "edit operation metrics, performance data, binary file attempt statistics, " +
          "default fuzzy matching parameters, concurrency settings, and configuration source. " +
          "Useful for monitoring, debugging, and performance analysis of server behavior.",
        inputSchema: {
          type: "object",
          properties: {},
          required: [],
        },
      },
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const operationTimer = new PerformanceTimer('request_handler');
  
  try {
    requestCount++;
    const { name, arguments: args } = request.params;

    logger.info({ tool: name, requestCount }, `Processing tool request: ${name}`);

    switch (name) {
      case "read_file": {
        const parsed = ReadFileArgsSchema.safeParse(args);
        if (!parsed.success) {
          throw createError('VALIDATION_ERROR', `Invalid arguments for read_file: ${parsed.error}`);
        }
        
        const validPath = await validatePath(parsed.data.path);
        const buffer = await fs.readFile(validPath);
        
        let content: string;
        let encoding: string;
        
        if (parsed.data.encoding === 'base64' || 
           (parsed.data.encoding === 'auto' && isBinaryFile(buffer, validPath))) {
          content = buffer.toString('base64');
          encoding = 'base64';
          logger.debug({ path: validPath, size: buffer.length }, 'Serving file as base64');
        } else {
          content = buffer.toString('utf-8');
          encoding = 'utf-8';
        }
        
        operationTimer.end({ tool: name, encoding, size: buffer.length });
        return {
          content: [{ 
            type: "text", 
            text: `File: ${parsed.data.path}\nEncoding: ${encoding}\nSize: ${buffer.length} bytes\n\n${content}` 
          }],
        };
      }

      case "read_multiple_files": {
        const parsed = ReadMultipleFilesArgsSchema.safeParse(args);
        if (!parsed.success) {
          throw createError('VALIDATION_ERROR', `Invalid arguments for read_multiple_files: ${parsed.error}`);
        }
        
        const results = await Promise.all(
          parsed.data.paths.map(async (filePath: string) => {
            try {
              const validPath = await validatePath(filePath);
              const buffer = await fs.readFile(validPath);
              
              let content: string;
              let encoding: string;
              
              if (parsed.data.encoding === 'base64' || 
                 (parsed.data.encoding === 'auto' && isBinaryFile(buffer, validPath))) {
                content = buffer.toString('base64');
                encoding = 'base64';
              } else {
                content = buffer.toString('utf-8');
                encoding = 'utf-8';
              }
              
              return `${filePath} (${encoding}, ${buffer.length} bytes):\n${content}\n`;
            } catch (error) {
              const structuredError = error as StructuredError;
              const errorMessage = structuredError.code ? 
                `${structuredError.code}: ${structuredError.message}` : 
                (error instanceof Error ? error.message : String(error));
              return `${filePath}: Error - ${errorMessage}`;
            }
          }),
        );
        
        operationTimer.end({ tool: name, filesCount: parsed.data.paths.length });
        return {
          content: [{ type: "text", text: results.join("\n---\n") }],
        };
      }

      case "write_file": {
        const parsed = WriteFileArgsSchema.safeParse(args);
        if (!parsed.success) {
          throw createError('VALIDATION_ERROR', `Invalid arguments for write_file: ${parsed.error}`);
        }
        
        const validPath = await validatePath(parsed.data.path);
        
        if (parsed.data.encoding === 'base64') {
          const buffer = Buffer.from(parsed.data.content, 'base64');
          await fs.writeFile(validPath, buffer);
          logger.info({ path: validPath, size: buffer.length, encoding: 'base64' }, 'File written');
        } else {
          await fs.writeFile(validPath, parsed.data.content, "utf-8");
          logger.info({ path: validPath, size: parsed.data.content.length, encoding: 'utf-8' }, 'File written');
        }
        
        operationTimer.end({ tool: name, encoding: parsed.data.encoding });
        return {
          content: [{ type: "text", text: `Successfully wrote to ${parsed.data.path}` }],
        };
      }

      case "edit_file": {
        editOperationCount++;
        const parsed = EditFileArgsSchema.safeParse(args);
        if (!parsed.success) {
          throw createError('VALIDATION_ERROR', `Invalid arguments for edit_file: ${parsed.error}`);
        }
        
        const validPath = await validatePath(parsed.data.path);
        
        // Use file-level locking for concurrent safety
        const fileLock = getFileLock(validPath);
        
        return await fileLock.runExclusive(async () => {
          const config: FuzzyMatchConfig = {
            maxDistanceRatio: parsed.data.maxDistanceRatio,
            minSimilarity: parsed.data.minSimilarity,
            caseSensitive: parsed.data.caseSensitive,
            ignoreWhitespace: parsed.data.ignoreWhitespace,
            preserveLeadingWhitespace: parsed.data.preserveLeadingWhitespace,
            debug: parsed.data.debug
          };

          const { modifiedContent, formattedDiff } = await applyFileEdits(
            validPath, 
            parsed.data.edits, 
            config
          );

          if (!parsed.data.dryRun) {
            await fs.writeFile(validPath, modifiedContent, 'utf-8');
            logger.info({ 
              path: validPath, 
              editsCount: parsed.data.edits.length,
              dryRun: false
            }, 'File edited successfully');
          } else {
            logger.info({ 
              path: validPath, 
              editsCount: parsed.data.edits.length,
              dryRun: true
            }, 'Dry run completed');
          }

          const resultMessage = parsed.data.dryRun 
            ? `Dry run completed. Here are the changes that would be made:\n\n${formattedDiff}`
            : `File edited successfully. Changes made:\n\n${formattedDiff}`;

          operationTimer.end({ tool: name, editsCount: parsed.data.edits.length, dryRun: parsed.data.dryRun });
          return {
            content: [{ type: "text", text: resultMessage }],
          };
        });
      }

      case "create_directory": {
        const parsed = CreateDirectoryArgsSchema.safeParse(args);
        if (!parsed.success) {
          throw createError('VALIDATION_ERROR', `Invalid arguments for create_directory: ${parsed.error}`);
        }
        
        const validPath = await validatePath(parsed.data.path);
        await fs.mkdir(validPath, { recursive: true });
        
        logger.info({ path: validPath }, 'Directory created');
        operationTimer.end({ tool: name });
        return {
          content: [{ type: "text", text: `Successfully created directory ${parsed.data.path}` }],
        };
      }

      case "list_directory": {
        const parsed = ListDirectoryArgsSchema.safeParse(args);
        if (!parsed.success) {
          throw createError('VALIDATION_ERROR', `Invalid arguments for list_directory: ${parsed.error}`);
        }
        
        const validPath = await validatePath(parsed.data.path);
        const entries = await fs.readdir(validPath, { withFileTypes: true });
        const formatted = entries
          .map((entry) => `${entry.isDirectory() ? "[DIR]" : "[FILE]"} ${entry.name}`)
          .join("\n");
        
        operationTimer.end({ tool: name, entriesCount: entries.length });
        return {
          content: [{ type: "text", text: formatted }],
        };
      }

      case "directory_tree": {
        const parsed = DirectoryTreeArgsSchema.safeParse(args);
        if (!parsed.success) {
          throw createError('VALIDATION_ERROR', `Invalid arguments for directory_tree: ${parsed.error}`);
        }

        interface TreeEntry {
          name: string;
          type: 'file' | 'directory';
          children?: TreeEntry[];
        }

        const visitedInodes = new Set<string>();

        async function buildTree(currentPath: string): Promise<TreeEntry[]> {
          const validPath = await validatePath(currentPath);
          
          try {
            const stats = await fs.stat(validPath);
            const inodeKey = `${stats.dev}-${stats.ino}`;
            
            if (visitedInodes.has(inodeKey)) {
              logger.debug({ path: currentPath }, 'Skipping already visited inode in tree build');
              return []; // Avoid infinite loops with symlinks
            }
            visitedInodes.add(inodeKey);

            const entries = await fs.readdir(validPath, {withFileTypes: true});
            const result: TreeEntry[] = [];

            for (const entry of entries) {
              const entryData: TreeEntry = {
                name: entry.name,
                type: entry.isDirectory() ? 'directory' : 'file'
              };

              if (entry.isDirectory()) {
                const subPath = path.join(currentPath, entry.name);
                entryData.children = await buildTree(subPath);
              }

              result.push(entryData);
            }

            visitedInodes.delete(inodeKey); // Clean up for this branch
            return result;
          } catch (error) {
            logger.debug({ path: currentPath, error: (error as Error).message }, 'Skipping inaccessible directory in tree build');
            return []; // Return empty array for inaccessible directories
          }
        }

        const treeData = await buildTree(parsed.data.path);
        operationTimer.end({ tool: name });
        return {
          content: [{
            type: "text",
            text: JSON.stringify(treeData, null, 2)
          }],
        };
      }

      case "move_file": {
        const parsed = MoveFileArgsSchema.safeParse(args);
        if (!parsed.success) {
          throw createError('VALIDATION_ERROR', `Invalid arguments for move_file: ${parsed.error}`);
        }
        
        const validSourcePath = await validatePath(parsed.data.source);
        const validDestPath = await validatePath(parsed.data.destination);
        await fs.rename(validSourcePath, validDestPath);
        
        logger.info({ source: validSourcePath, destination: validDestPath }, 'File moved');
        operationTimer.end({ tool: name });
        return {
          content: [{ type: "text", text: `Successfully moved ${parsed.data.source} to ${parsed.data.destination}` }],
        };
      }

      case "search_files": {
        const parsed = SearchFilesArgsSchema.safeParse(args);
        if (!parsed.success) {
          throw createError('VALIDATION_ERROR', `Invalid arguments for search_files: ${parsed.error}`);
        }
        
        const validPath = await validatePath(parsed.data.path);
        const results = await searchFiles(
          validPath, 
          parsed.data.pattern, 
          parsed.data.excludePatterns,
          parsed.data.useExactPatterns
        );
        
        operationTimer.end({ tool: name, resultsCount: results.length });
        return {
          content: [{ type: "text", text: results.length > 0 ? results.join("\n") : "No matches found" }],
        };
      }

      case "get_file_info": {
        const parsed = GetFileInfoArgsSchema.safeParse(args);
        if (!parsed.success) {
          throw createError('VALIDATION_ERROR', `Invalid arguments for get_file_info: ${parsed.error}`);
        }
        
        const validPath = await validatePath(parsed.data.path);
        const info = await getFileStats(validPath);
        
        operationTimer.end({ tool: name, isBinary: info.isBinary });
        return {
          content: [{ type: "text", text: Object.entries(info)
            .map(([key, value]) => `${key}: ${value}`)
            .join("\n") }],
        };
      }

      case "list_allowed_directories": {
        operationTimer.end({ tool: name });
        return {
          content: [{
            type: "text",
            text: `Allowed directories:\n${allowedDirectories.join('\n')}`
          }],
        };
      }

      case "server_stats": {
        const isConfigFile = args.length > 0 && (args[0] === '--config' || args[0] === '-c');
        const stats = {
          requests_handled: requestCount,
          edit_operations: editOperationCount,
          binary_file_attempts: binaryFileAttempts,
          average_edit_time_ms: Math.round(averageEditTime * 100) / 100,
          configuration: {
            fuzzy_matching: config.fuzzyMatching,
            logging: config.logging,
            concurrency: config.concurrency,
            source: isConfigFile ? 'config_file' : 'cli_args'
          },
          allowed_directories: allowedDirectories,
          active_file_locks: fileLocks.size,
          server_version: "0.5.0",
          uptime_ms: process.uptime() * 1000
        };
        
        operationTimer.end({ tool: name });
        return {
          content: [{
            type: "text",
            text: JSON.stringify(stats, null, 2)
          }],
        };
      }

      default:
        throw createError('UNKNOWN_TOOL', `Unknown tool: ${name}`, { toolName: name });
    }
  } catch (error) {
    operationTimer.end({ result: 'error' });
    
    let structuredError: StructuredError;
    if ((error as any).code) {
      structuredError = error as StructuredError;
    } else {
      structuredError = createError(
        'UNKNOWN_ERROR',
        error instanceof Error ? error.message : String(error)
      );
    }
    
    logger.error({
      error: structuredError,
      tool: request.params.name,
      requestCount
    }, `Tool request failed: ${request.params.name}`);
    
    return {
      content: [{ 
        type: "text", 
        text: `Error (${structuredError.code}): ${structuredError.message}${
          structuredError.details ? `\nDetails: ${JSON.stringify(structuredError.details, null, 2)}` : ''
        }` 
      }],
      isError: true,
    };
  }
});

// Start server
async function runServer() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  
  logger.info({
    version: '0.5.0',
    allowedDirectories,
    config: {
      fuzzyMatching: config.fuzzyMatching,
      logging: config.logging,
      concurrency: config.concurrency
    }
  }, "Enhanced MCP Filesystem Server with Advanced Features started");
  
  console.error("Enhanced MCP Filesystem Server with Advanced Features v0.5.0 running on stdio");
  console.error("Allowed directories:", allowedDirectories);
  console.error("Logging level:", config.logging.level);
}

runServer().catch((error) => {
  logger.fatal({ error }, "Fatal error running server");
  console.error("Fatal error running server:", error);
  process.exit(1);
});