import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { Mutex } from 'async-mutex';
import fs from 'node:fs/promises';
import path from 'node:path';

import { createError, StructuredError } from '../types/errors.js';
import { PerformanceTimer } from '../utils/performance.js';
import { isBinaryFile } from '../utils/binaryDetect.js';
import { validatePath } from './security.js';
import { applyFileEdits, FuzzyMatchConfig } from './fuzzyEdit.js';
import { getFileStats, searchFiles, readMultipleFilesContent, FileReadResult, getDirectoryTree } from './fileInfo.js'; 
import * as schemas from './schemas.js';
// Import specific types that were causing issues if not directly imported
import type { ListDirectoryEntry, DirectoryTreeEntry, EditOperation } from './schemas.js'; 

import type { Logger } from 'pino';
import type { Config } from '../server/config.js';

let requestCount = 0;
let editOperationCount = 0;

const fileLocks = new Map<string, Mutex>();

function getFileLock(filePath: string, config: Config): Mutex {
  if (!fileLocks.has(filePath)) {
    if (fileLocks.size >= config.concurrency.maxConcurrentEdits) {
      const oldestKey = fileLocks.keys().next().value;
      if (oldestKey) fileLocks.delete(oldestKey);
    }
    fileLocks.set(filePath, new Mutex());
  }
  return fileLocks.get(filePath)!;
}

export function setupToolHandlers(server: Server, allowedDirectories: string[], logger: Logger, config: Config) {
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
          return { result: allowedDirectories };
        }

        case 'read_file': {
          const parsed = schemas.ReadFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ReadFileArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          const rawBuffer = await fs.readFile(validPath);
          let content: string;
          let encodingUsed: 'utf-8' | 'base64' = 'utf-8';
          const isBinary = isBinaryFile(rawBuffer, validPath);

          if (parsed.data.encoding === 'base64' || (parsed.data.encoding === 'auto' && isBinary)) {
            content = rawBuffer.toString('base64');
            encodingUsed = 'base64';
          } else {
            content = rawBuffer.toString('utf-8'); // Default to utf-8
          }
          return { result: { content, encoding: encodingUsed } };
        }

        case 'read_multiple_files': {
            const parsed = schemas.ReadMultipleFilesArgsSchema.safeParse(request.params.args);
            if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ReadMultipleFilesArgsSchema) });

            // const timer = new PerformanceTimer('read_multiple_files_handler', logger, config); // Timer is now within readMultipleFilesContent
            const fileReadResults = await readMultipleFilesContent(
              parsed.data.paths,
              parsed.data.encoding,
              allowedDirectories,
              logger,
              config
            );
            // timer.end(...); // Logging for timer is handled within readMultipleFilesContent
            return { result: fileReadResults };
        }

        case 'write_file': {
          const parsed = schemas.WriteFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.WriteFileArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config); // isWriteOperation = true

          const lock = getFileLock(validPath, config);
          await lock.runExclusive(async () => {
            const contentBuffer = Buffer.from(parsed.data.content, parsed.data.encoding);
            await fs.writeFile(validPath, contentBuffer);
          });
          return { content: [{ type: 'text', text: `File written: ${parsed.data.path}` }] };
        }

        case 'edit_file': {
          editOperationCount++;
          const parsed = EditFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments for edit_file', { error: parsed.error, schema: zodToJsonSchema(EditFileArgsSchema) });
          
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config); // isWriteOperation = true
          const fileBuffer = await fs.readFile(validPath);
          if (isBinaryFile(fileBuffer, validPath)) { // Check if file is binary by extension or content
            throw createError('BINARY_FILE_EDIT', `Cannot edit binary file: ${validPath}`);
          }

          const fuzzyConfig: FuzzyMatchConfig = {
            maxDistanceRatio: parsed.data.maxDistanceRatio,
            minSimilarity: parsed.data.minSimilarity,
            caseSensitive: parsed.data.caseSensitive,
            ignoreWhitespace: parsed.data.ignoreWhitespace,
            preserveLeadingWhitespace: parsed.data.preserveLeadingWhitespace,
            debug: parsed.data.debug
          };
          
          let modifiedContent: string = '';
          let formattedDiff: string = '';

          const lock = getFileLock(validPath, config);
          await lock.runExclusive(async () => {
            const currentContent = await fs.readFile(validPath, 'utf-8');
            const editResult = await applyFileEdits(currentContent, parsed.data.edits as EditOperation[], fuzzyConfig, logger, config);
            modifiedContent = editResult.modifiedContent;
            formattedDiff = editResult.formattedDiff;
            if (!parsed.data.dryRun) {
              await fs.writeFile(validPath, modifiedContent, 'utf-8');
            }
          });

          const responseText = parsed.data.dryRun 
            ? `Dry run: File '${parsed.data.path}' would be modified. Diff:\n${formattedDiff}`
            : `File '${parsed.data.path}' edited successfully. Diff:\n${formattedDiff}`;
          
          return { content: [{ type: 'text', text: responseText }] };
        }
        
        case 'create_directory': {
          const parsed = schemas.CreateDirectoryArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.CreateDirectoryArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config); // isWriteOperation = true
          await fs.mkdir(validPath, { recursive: true });
          return { content: [{ type: 'text', text: `Directory created: ${parsed.data.path}` }] };
        }

        case 'list_directory': {
          const parsed = schemas.ListDirectoryArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ListDirectoryArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          const entries = await fs.readdir(validPath, { withFileTypes: true });
          const results: ListDirectoryEntry[] = [];
          for (const dirent of entries) {
            let type: ListDirectoryEntry['type'] = 'other';
            if (dirent.isFile()) type = 'file';
            else if (dirent.isDirectory()) type = 'directory';
            else if (dirent.isSymbolicLink()) type = 'symlink';
            
            const entryPath = path.join(validPath, dirent.name);
            let size: number | undefined = undefined;
            if (type === 'file') {
                try {
                    const stats = await fs.stat(entryPath);
                    size = stats.size;
                } catch (statError) {
                    logger.warn({ path: entryPath, error: statError }, 'Failed to get stats for file in list_directory');
                }
            }
            results.push({ name: dirent.name, path: path.relative(config.allowedDirectories[0], entryPath).replace(/\\/g, '/'), type, size });
          }
          return { result: results };
        }

        case 'move_file': {
          const parsed = schemas.MoveFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.MoveFileArgsSchema) });
          const validSource = await validatePath(parsed.data.source, allowedDirectories, logger, config); // isWriteOperation = true (on source)
          const validDestination = await validatePath(parsed.data.destination, allowedDirectories, logger, config); // isWriteOperation = true (on destination)
          
          const sourceLock = getFileLock(validSource, config);
          const destLock = getFileLock(validDestination, config); // Potentially lock destination too if it might exist or be created
          
          await sourceLock.runExclusive(async () => {
            // If destination is different, also acquire its lock if not already held
            if (validSource !== validDestination && !destLock.isLocked()) {
              await destLock.runExclusive(async () => {
                await fs.rename(validSource, validDestination);
              });
            } else {
              // If source and destination are same or destLock already acquired (e.g. by sourceLock if paths are same)
              await fs.rename(validSource, validDestination);
            }
          });
          return { content: [{ type: 'text', text: `Moved from ${parsed.data.source} to ${parsed.data.destination}` }] };
        }

        case 'delete_file': {
          const parsed = schemas.DeleteFileArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.DeleteFileArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config); // isWriteOperation = true
          
          const lock = getFileLock(validPath, config);
          await lock.runExclusive(async () => {
            await fs.unlink(validPath);
          });
          return { content: [{ type: 'text', text: `File deleted: ${parsed.data.path}` }] };
        }

        case 'delete_directory': {
          const parsed = schemas.DeleteDirectoryArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.DeleteDirectoryArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config); // isWriteOperation = true
          
          const lock = getFileLock(validPath, config); // Lock the directory itself
          await lock.runExclusive(async () => {
            await fs.rm(validPath, { recursive: parsed.data.recursive || false, force: false }); // force: false for safety
          });
          return { content: [{ type: 'text', text: `Directory deleted: ${parsed.data.path}` }] };
        }

        case 'search_files': {
          const parsed = schemas.SearchFilesArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.SearchFilesArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          const results = await searchFiles(validPath, parsed.data.pattern, logger, config, parsed.data.excludePatterns || [], false);
          return { result: results };
        }

        case 'get_file_info': {
          const parsed = schemas.GetFileInfoArgsSchema.safeParse(request.params.args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.GetFileInfoArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          const stats = await getFileStats(validPath, logger, config);
          return { result: stats };
        }

        case 'directory_tree': {
            const parsed = schemas.DirectoryTreeArgsSchema.safeParse(request.params.args);
            if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.DirectoryTreeArgsSchema) });
            const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
            const tree = await getDirectoryTree(validPath, allowedDirectories, logger, config);
            logger.debug({ tree }, 'Generated directory tree (getDirectoryTree)');
            return { content: [{ type: 'text', text: JSON.stringify(tree, null, 2) }] };
        }

        case 'server_stats': {
            const parsed = schemas.ServerStatsArgsSchema.safeParse(request.params.args);
            if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ServerStatsArgsSchema) });
            const stats = { requestCount, editOperationCount, config };
            return { content: [{ type: 'text', text: JSON.stringify(stats, null, 2) }] };
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
