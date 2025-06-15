import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { Mutex } from 'async-mutex';
import { initGlobalSemaphore, getGlobalSemaphore } from './concurrency.js';
import fs from 'node:fs/promises';
import path from 'node:path';
import { minimatch } from 'minimatch';

import { createError, StructuredError } from '../types/errors.js';
import { PerformanceTimer } from '../utils/performance.js';
import { isBinaryFile } from '../utils/binaryDetect.js';
import { validatePath } from './security.js';
import { applyFileEdits, FuzzyMatchConfig } from './fuzzyEdit.js';
import { getFileStats, searchFiles, readMultipleFilesContent, getDirectoryTree } from './fileInfo.js';
import * as schemas from './schemas.js'; // <-- BRAKUJĄCY IMPORT ZOSTAŁ DODANY
// Import specific types that were causing issues if not directly imported
import type { ListDirectoryEntry, DirectoryTreeEntry } from './schemas.js';

import type { Logger } from 'pino';
import type { Config } from '../server/config.js';

let requestCount = 0;
let editOperationCount = 0;

const fileLocks = new Map<string, Mutex>();

function getFileLock(filePath: string, config: Config, logger: Logger): Mutex {
  if (fileLocks.has(filePath)) {
    // Move to end of map to mark as recently used
    const existingMutex = fileLocks.get(filePath)!;
    fileLocks.delete(filePath);
    fileLocks.set(filePath, existingMutex);
    return existingMutex;
  } else {
    if (fileLocks.size >= config.concurrency.maxConcurrentEdits) {
      let evicted = false;
      // Iterate from oldest (insertion order)
      for (const [key, mutex] of fileLocks.entries()) {
        if (!mutex.isLocked()) {
          fileLocks.delete(key);
          logger.debug({ evictedKey: key, newKey: filePath, mapSize: fileLocks.size }, 'Evicted inactive lock to make space.');
          evicted = true;
          break;
        }
      }
      if (!evicted) {
        // All locks are active, and map is full
        logger.error({ filePath, mapSize: fileLocks.size, maxConcurrentEdits: config.concurrency.maxConcurrentEdits }, 'Cannot acquire new file lock: max concurrent locks reached, and all are active.');
        throw createError('MAX_CONCURRENCY_REACHED', `Cannot acquire new file lock for ${filePath}: Maximum concurrent file locks (${config.concurrency.maxConcurrentEdits}) reached, and all are currently active.`);
      }
    }
    const newMutex = new Mutex();
    fileLocks.set(filePath, newMutex);
    return newMutex;
  }
}

function shouldSkipPath(filePath: string, config: Config): boolean {
  // Use the first allowed directory as the base for relative path calculation
  const baseDir = config.allowedDirectories[0] ?? process.cwd();
  const relativePath = path.relative(baseDir, filePath);

  // Check against default excludes
  if (config.fileFiltering.defaultExcludes.some(p => minimatch(relativePath, p, { dot: true }))) {
    return true;
  }

  // Check allowed extensions if forceTextFiles is true
  if (config.fileFiltering.forceTextFiles) {
    const ext = path.extname(filePath).toLowerCase();
    const isAllowed = config.fileFiltering.allowedExtensions.some(p => minimatch(`*${ext}`, p, { nocase: true }));
    if (!isAllowed) {
      return true;
    }
  }

  return false;
}

export function setupToolHandlers(server: Server, allowedDirectories: string[], logger: Logger, config: Config) {
  initGlobalSemaphore(config);
  server.setRequestHandler(schemas.HandshakeRequestSchema, async () => ({
    serverName: 'mcp-filesystem-server',
    serverVersion: '0.7.0'
  }));
  const EditFileArgsSchema = schemas.getEditFileArgsSchema(config);

  server.setRequestHandler(schemas.ListToolsRequestSchema, async () => {
    return {
      tools: [
        { name: 'read_file', description: 'Read file contents.', inputSchema: zodToJsonSchema(schemas.ReadFileArgsSchema) as any },
        { name: 'read_multiple_files', description: 'Read multiple files.', inputSchema: zodToJsonSchema(schemas.ReadMultipleFilesArgsSchema) as any },
        { name: 'list_allowed_directories', description: 'List allowed base directories.', inputSchema: zodToJsonSchema(schemas.ListAllowedDirectoriesArgsSchema) as any },
        { name: 'write_file', description: 'Write file contents.', inputSchema: zodToJsonSchema(schemas.WriteFileArgsSchema) as any },
        { name: 'edit_file', description: 'Edit file contents using fuzzy matching.', inputSchema: zodToJsonSchema(EditFileArgsSchema) as any },
        { name: 'create_directory', description: 'Create a directory.', inputSchema: zodToJsonSchema(schemas.CreateDirectoryArgsSchema) as any },
        { name: 'list_directory', description: 'List directory contents.', inputSchema: zodToJsonSchema(schemas.ListDirectoryArgsSchema) as any },
        { name: 'directory_tree', description: 'Get directory tree.', inputSchema: zodToJsonSchema(schemas.DirectoryTreeArgsSchema) as any },
        { name: 'move_file', description: 'Move/rename a file or directory.', inputSchema: zodToJsonSchema(schemas.MoveFileArgsSchema) as any },
        { name: 'delete_file', description: 'Delete a file.', inputSchema: zodToJsonSchema(schemas.DeleteFileArgsSchema) as any },
        { name: 'delete_directory', description: 'Delete a directory.', inputSchema: zodToJsonSchema(schemas.DeleteDirectoryArgsSchema) as any },
        { name: 'search_files', description: 'Search for files by pattern.', inputSchema: zodToJsonSchema(schemas.SearchFilesArgsSchema) as any },
        { name: 'get_file_info', description: 'Get file/directory metadata.', inputSchema: zodToJsonSchema(schemas.GetFileInfoArgsSchema) as any },
        { name: 'server_stats', description: 'Get server statistics.', inputSchema: zodToJsonSchema(schemas.ServerStatsArgsSchema) as any }
      ]
    };
  });

  server.setRequestHandler(schemas.CallToolRequestSchema, async (request) => {
    requestCount++;
    logger.info({ tool: request.params.name, args: request.params.args }, `Tool request: ${request.params.name}`);
    try {
      switch (request.params.name) {
        case 'list_allowed_directories': {
          const parsed = schemas.ListAllowedDirectoriesArgsSchema.safeParse(request.params.args ?? {});
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ListAllowedDirectoriesArgsSchema) });
          return { result: { directories: allowedDirectories } };
        }

        case 'read_file': {
          const parsed = schemas.ReadFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ReadFileArgsSchema) });
          const timer = new PerformanceTimer('read_file_handler', logger, config);
          try {
            const validatedPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
            if (shouldSkipPath(validatedPath, config)) {
              throw createError('ACCESS_DENIED', 'File access denied due to filtering rules.');
            }
            const rawBuffer = await fs.readFile(validatedPath);
            let content: string;
            let encodingUsed: 'utf-8' | 'base64' = 'utf-8';
            const isBinary = isBinaryFile(rawBuffer, validatedPath);

            if (parsed.data.encoding === 'base64' || (parsed.data.encoding === 'auto' && isBinary)) {
              content = rawBuffer.toString('base64');
              encodingUsed = 'base64';
            } else {
              content = rawBuffer.toString('utf-8'); // Default to utf-8
            }
            timer.end();
            return { result: { content, encoding: encodingUsed } };
          } catch (error) {
            timer.end({ result: 'error' });
            throw error;
          }
        }

        case 'read_multiple_files': {
          const parsed = schemas.ReadMultipleFilesArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ReadMultipleFilesArgsSchema) });
          const fileReadResults = await readMultipleFilesContent(
            parsed.data.paths,
            parsed.data.encoding,
            allowedDirectories,
            logger,
            config
          );
          return { result: { files: fileReadResults } };
        }

        case 'write_file': {
          const parsed = schemas.WriteFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.WriteFileArgsSchema) });
          const timer = new PerformanceTimer('write_file_handler', logger, config);
          try {
            const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
            if (shouldSkipPath(validPath, config)) {
              throw createError('ACCESS_DENIED', 'File writing denied due to filtering rules.');
            }
            const lock = getFileLock(validPath, config, logger);
            await getGlobalSemaphore().runExclusive(async () => {
              await lock.runExclusive(async () => {
                const contentBuffer = Buffer.from(parsed.data.content, parsed.data.encoding);
                await fs.writeFile(validPath, contentBuffer);
              });
            });
            timer.end();
            return { content: [{ type: 'text', text: `File written: ${parsed.data.path}` }] };
          } catch (error) {
            timer.end({ result: 'error' });
            throw error;
          }
        }

        case 'edit_file': {
          editOperationCount++;
          const parsed = EditFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          if (shouldSkipPath(validPath, config)) {
            throw createError('ACCESS_DENIED', 'File editing denied due to filtering rules', { path: validPath });
          }
          const fuzzyConfig: FuzzyMatchConfig = {
            maxDistanceRatio: parsed.data.maxDistanceRatio,
            minSimilarity: parsed.data.minSimilarity,
            caseSensitive: parsed.data.caseSensitive,
            ignoreWhitespace: parsed.data.ignoreWhitespace,
            preserveLeadingWhitespace: parsed.data.preserveLeadingWhitespace,
            debug: parsed.data.debug || config.logging.level === 'debug',
          };

          const lock = getFileLock(validPath, config, logger);
          let formattedDiff: string = '';

          await getGlobalSemaphore().runExclusive(async () => {
            await lock.runExclusive(async () => {
              const editResult = await applyFileEdits(
                validPath,
                parsed.data.edits,
                fuzzyConfig,
                logger,
                config
              );
              formattedDiff = editResult.formattedDiff;

              if (!parsed.data.dryRun) {
                await fs.writeFile(validPath, editResult.modifiedContent, 'utf-8');
              }
            });
          });

          const responseText = parsed.data.dryRun
            ? `Dry run: File '${parsed.data.path}' would be modified. Diff:\n${formattedDiff}`
            : `File '${parsed.data.path}' edited successfully. Diff:\n${formattedDiff}`;

          return { content: [{ type: 'text', text: responseText }] };
        }

        case 'create_directory': {
          const parsed = schemas.CreateDirectoryArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.CreateDirectoryArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          const lock = getFileLock(validPath, config, logger);
          await getGlobalSemaphore().runExclusive(async () => {
            await lock.runExclusive(async () => {
              await fs.mkdir(validPath, { recursive: true });
            });
          });
          return { content: [{ type: 'text', text: `Directory created: ${parsed.data.path}` }] };
        }

        case 'list_directory': {
          const parsed = schemas.ListDirectoryArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ListDirectoryArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          const entries = await fs.readdir(validPath, { withFileTypes: true });
          const results: ListDirectoryEntry[] = [];
          for (const dirent of entries) {
            const entryPath = path.join(validPath, dirent.name);
            if (shouldSkipPath(entryPath, config)) {
              continue;
            }

            let type: ListDirectoryEntry['type'] = 'other';
            if (dirent.isFile()) type = 'file';
            else if (dirent.isDirectory()) type = 'directory';
            else if (dirent.isSymbolicLink()) type = 'symlink';

            let size: number | undefined = undefined;
            if (type === 'file') {
              try {
                const stats = await fs.stat(entryPath);
                size = stats.size;
              } catch (statError) {
                logger.warn({ path: entryPath, error: statError }, 'Failed to get stats for file in list_directory');
              }
            }
            results.push({ name: dirent.name, path: path.relative(allowedDirectories[0], entryPath).replace(/\\/g, '/'), type, size });
          }
          return { result: { entries: results } };
        }

        case 'move_file': {
          const parsed = schemas.MoveFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.MoveFileArgsSchema) });

          const validSource = await validatePath(parsed.data.source, allowedDirectories, logger, config);
          const validDestination = await validatePath(parsed.data.destination, allowedDirectories, logger, config);

          if (validSource === validDestination) {
            return { content: [{ type: 'text', text: 'Source and destination are the same, no action taken.' }] };
          }
          if (shouldSkipPath(validSource, config) || shouldSkipPath(validDestination, config)) {
            throw createError('ACCESS_DENIED', 'Source or destination path is disallowed by filtering rules.');
          }

          await getGlobalSemaphore().runExclusive(async () => {
            // To prevent deadlocks, always acquire locks in a consistent order (alphabetical)
            const [path1, path2] = [validSource, validDestination].sort();
            const lock1 = getFileLock(path1, config, logger);
            const lock2 = getFileLock(path2, config, logger);

            await lock1.runExclusive(async () => {
              await lock2.runExclusive(async () => {
                await fs.rename(validSource, validDestination);
              });
            });
          });

          return { content: [{ type: 'text', text: `Moved from ${parsed.data.source} to ${parsed.data.destination}` }] };
        }

        case 'delete_file': {
          const parsed = schemas.DeleteFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.DeleteFileArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          if (shouldSkipPath(validPath, config)) {
            throw createError('ACCESS_DENIED', 'File deletion denied due to filtering rules.');
          }
          const lock = getFileLock(validPath, config, logger);
          await getGlobalSemaphore().runExclusive(async () => {
            await lock.runExclusive(async () => {
              await fs.unlink(validPath);
            });
          });
          return { content: [{ type: 'text', text: `File deleted: ${parsed.data.path}` }] };
        }

        case 'delete_directory': {
          const parsed = schemas.DeleteDirectoryArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.DeleteDirectoryArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          if (shouldSkipPath(validPath, config)) {
            throw createError('ACCESS_DENIED', 'Directory deletion denied due to filtering rules.');
          }
          const lock = getFileLock(validPath, config, logger);
          await getGlobalSemaphore().runExclusive(async () => {
            await lock.runExclusive(async () => {
              await fs.rm(validPath, { recursive: parsed.data.recursive || false, force: false }); // force: false for safety
            });
          });
          return { content: [{ type: 'text', text: `Directory deleted: ${parsed.data.path}` }] };
        }

        case 'search_files': {
          const parsed = schemas.SearchFilesArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.SearchFilesArgsSchema) });
          const timer = new PerformanceTimer('search_files_handler', logger, config);
          try {
            const validatedPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
            const results = await searchFiles(
              validatedPath,
              parsed.data.pattern,
              logger,
              config,
              parsed.data.excludePatterns || [],
              parsed.data.useExactPatterns || false,
              parsed.data.maxDepth || -1
            );

            let finalResults = results;
            if (parsed.data.maxResults && results.length > parsed.data.maxResults) {
              finalResults = results.slice(0, parsed.data.maxResults);
            }

            timer.end({ resultsCount: finalResults.length });
            return { result: { paths: finalResults } };
          } catch (error) {
            timer.end({ result: 'error' });
            throw error;
          }
        }

        case 'get_file_info': {
          const parsed = schemas.GetFileInfoArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.GetFileInfoArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          if (shouldSkipPath(validPath, config)) {
            throw createError('ACCESS_DENIED', 'File access denied due to filtering rules.');
          }
          const stats = await getFileStats(validPath, logger, config);
          return { result: stats };
        }

        case 'directory_tree': {
          const parsed = schemas.DirectoryTreeArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.DirectoryTreeArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          const tree = await getDirectoryTree(validPath, allowedDirectories, logger, config, 0, parsed.data.maxDepth ?? -1);
          return { result: tree };
        }

        case 'server_stats': {
          const parsed = schemas.ServerStatsArgsSchema.safeParse(request.params.args ?? {});
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ServerStatsArgsSchema) });
          const stats = { requestCount, editOperationCount, activeLocks: fileLocks.size, config };
          return { result: stats };
        }

        default:
          throw createError('UNKNOWN_TOOL', `Unknown tool: ${request.params.name}`);
      }
    } catch (error) {
      let structuredError: StructuredError;
      if ((error as any).code && (error as any).message) {
        structuredError = error as StructuredError;
      } else {
        structuredError = createError('UNKNOWN_ERROR', error instanceof Error ? error.message : String(error));
      }
      logger.error({ error: structuredError, tool: request.params.name }, `Tool request failed: ${request.params.name}`);
      return {
        content: [{ type: 'text', text: `Error (${structuredError.code}): ${structuredError.message}` }],
        isError: true,
        meta: { hint: structuredError.hint, confidence: structuredError.confidence, details: structuredError.details }
      };
    }
  });
}