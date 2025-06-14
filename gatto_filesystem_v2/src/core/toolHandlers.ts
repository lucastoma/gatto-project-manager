import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { Mutex } from 'async-mutex';
import fs from 'node:fs/promises';
import path from 'node:path';

import { createError, StructuredError } from '../types/errors.js';
import { PerformanceTimer } from '../utils/performance.js';
import { isBinaryFile } from '../utils/binaryDetect.js';
import { validatePath } from './security.js';
import { applyFileEdits, FuzzyMatchConfig } from './fuzzyEdit.js';
import { getFileStats, searchFiles } from './fileInfo.js';
import * as schemas from './schemas.js';

import type { Logger } from 'pino';
import type { Config } from '../server/config.js';

let requestCount = 0;
let editOperationCount = 0;

const fileLocks = new Map<string, Mutex>();

function getFileLock(filePath: string, config: Config): Mutex {
  if (!fileLocks.has(filePath)) {
    if (fileLocks.size >= config.concurrency.maxConcurrentEdits) {
      const oldestKey = fileLocks.keys().next().value;
      fileLocks.delete(oldestKey);
    }
    fileLocks.set(filePath, new Mutex());
  }
  return fileLocks.get(filePath)!;
}

export function setupToolHandlers(server: Server, allowedDirectories: string[], logger: Logger, config: Config) {
  const EditFileArgsSchema = schemas.getEditFileArgsSchema(config);

  server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
      tools: [
        { name: 'read_file', description: 'Read file contents.', inputSchema: zodToJsonSchema(schemas.ReadFileArgsSchema) as any },
        { name: 'read_multiple_files', description: 'Read multiple files.', inputSchema: zodToJsonSchema(schemas.ReadMultipleFilesArgsSchema) as any },
        { name: 'write_file', description: 'Write to a file.', inputSchema: zodToJsonSchema(schemas.WriteFileArgsSchema) as any },
        { name: 'edit_file', description: 'Apply fuzzy edits to a file.', inputSchema: zodToJsonSchema(EditFileArgsSchema) as any },
        { name: 'create_directory', description: 'Create a directory.', inputSchema: zodToJsonSchema(schemas.CreateDirectoryArgsSchema) as any },
        { name: 'list_directory', description: 'List directory contents.', inputSchema: zodToJsonSchema(schemas.ListDirectoryArgsSchema) as any },
        { name: 'directory_tree', description: 'Get a directory tree.', inputSchema: zodToJsonSchema(schemas.DirectoryTreeArgsSchema) as any },
        { name: 'move_file', description: 'Move/rename a file.', inputSchema: zodToJsonSchema(schemas.MoveFileArgsSchema) as any },
        { name: 'search_files', description: 'Search for files.', inputSchema: zodToJsonSchema(schemas.SearchFilesArgsSchema) as any },
        { name: 'get_file_info', description: 'Get file metadata.', inputSchema: zodToJsonSchema(schemas.GetFileInfoArgsSchema) as any },
        { name: 'list_allowed_directories', description: 'List allowed directories.', inputSchema: { type: 'object', properties: {} } },
        { name: 'server_stats', description: 'Get server statistics.', inputSchema: { type: 'object', properties: {} } },
      ],
    };
  });

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const operationTimer = new PerformanceTimer('request_handler', logger, config);
    requestCount++;
    const { name, arguments: args } = request.params;
    logger.info({ tool: name, requestCount }, `Processing tool request: ${name}`);

    try {
      switch (name) {
        case 'read_file': {
          const parsed = schemas.ReadFileArgsSchema.safeParse(args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ReadFileArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          const buffer = await fs.readFile(validPath);
          let content: string, encoding: string;
          if (parsed.data.encoding === 'base64' || (parsed.data.encoding === 'auto' && isBinaryFile(buffer, validPath))) {
            content = buffer.toString('base64');
            encoding = 'base64';
          } else {
            content = buffer.toString('utf-8');
            encoding = 'utf-8';
          }
          return { content: [{ type: 'text', text: `File: ${validPath}\nEncoding: ${encoding}\n\n${content}` }] };
        }

        case 'read_multiple_files': {
          const parsed = schemas.ReadMultipleFilesArgsSchema.safeParse(args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ReadMultipleFilesArgsSchema) });
          const results: Array<{ path: string; content: string; encoding: string }> = [];
          for (const p of parsed.data.paths) {
            try {
              const validPath = await validatePath(p, allowedDirectories, logger, config);
              const buffer = await fs.readFile(validPath);
              const enc = (parsed.data.encoding === 'base64' || (parsed.data.encoding === 'auto' && isBinaryFile(buffer, validPath))) ? 'base64' : 'utf-8';
              const content = buffer.toString(enc as BufferEncoding);
              results.push({ path: validPath, content, encoding: enc });
            } catch (e) {
              results.push({ path: p, content: `Error: ${(e as Error).message}`, encoding: 'error' });
            }
          }
          return { content: [{ type: 'text', text: JSON.stringify(results, null, 2) }] };
        }

        case 'write_file': {
          const parsed = schemas.WriteFileArgsSchema.safeParse(args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.WriteFileArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          const buffer = Buffer.from(parsed.data.content, parsed.data.encoding);
          await fs.writeFile(validPath, buffer);
          return { content: [{ type: 'text', text: `Successfully wrote to ${validPath}` }] };
        }

        case 'edit_file': {
          editOperationCount++;
          const parsed = EditFileArgsSchema.safeParse(args);
          if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(EditFileArgsSchema) });
          const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
          const fileLock = getFileLock(validPath, config);
          return await fileLock.runExclusive(async () => {
            const { modifiedContent, formattedDiff } = await applyFileEdits(validPath, parsed.data.edits, parsed.data, logger, config);
            if (!parsed.data.dryRun) {
              await fs.writeFile(validPath, modifiedContent, 'utf-8');
            }
            const resultMessage = parsed.data.dryRun ? `Dry run complete. Changes:\n\n${formattedDiff}` : `File edited. Changes:\n\n${formattedDiff}`;
            return { content: [{ type: 'text', text: resultMessage }] };
          });
        }

        case 'create_directory': {
            const parsed = schemas.CreateDirectoryArgsSchema.safeParse(args);
            if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.CreateDirectoryArgsSchema) });
            const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
            await fs.mkdir(validPath, { recursive: true });
            return { content: [{ type: 'text', text: `Successfully created directory ${validPath}` }] };
        }

        case 'directory_tree': {
            const parsed = schemas.DirectoryTreeArgsSchema.safeParse(args);
            if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.DirectoryTreeArgsSchema) });
            const rootPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
            async function buildTree(currentPath: string): Promise<any> {
              const stats = await fs.stat(currentPath);
              const node: any = { name: path.basename(currentPath), type: stats.isDirectory() ? 'directory' : 'file' };
              if (stats.isDirectory()) {
                const children = await fs.readdir(currentPath);
                node.children = await Promise.all(children.map(child => buildTree(path.join(currentPath, child))));
              }
              return node;
            }
            const tree = await buildTree(rootPath);
            return { content: [{ type: 'text', text: JSON.stringify(tree, null, 2) }] };
        }

        case 'move_file': {
            const parsed = schemas.MoveFileArgsSchema.safeParse(args);
            if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.MoveFileArgsSchema) });
            const sourcePath = await validatePath(parsed.data.source, allowedDirectories, logger, config);
            const destPath = await validatePath(parsed.data.destination, allowedDirectories, logger, config);
            try {
              await fs.access(destPath);
              throw createError('DEST_EXISTS', 'Destination already exists', { destination: destPath });
            } catch {
              // dest does not exist – ok
            }
            await fs.rename(sourcePath, destPath);
            return { content: [{ type: 'text', text: `Moved ${sourcePath} -> ${destPath}` }] };
        }

        case 'list_directory': {
            const parsed = schemas.ListDirectoryArgsSchema.safeParse(args);
            if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.ListDirectoryArgsSchema) });
            const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
            const entries = await fs.readdir(validPath, { withFileTypes: true });
            const listing = entries.map(e => e.isDirectory() ? `[DIR]  ${e.name}` : `[FILE] ${e.name}`).join('\n');
            return { content: [{ type: 'text', text: listing }] };
        }

        case 'get_file_info': {
            const parsed = schemas.GetFileInfoArgsSchema.safeParse(args);
            if (!parsed.success) throw createError('VALIDATION_ERROR', 'Invalid arguments', { error: parsed.error, schema: zodToJsonSchema(schemas.GetFileInfoArgsSchema) });
            const validPath = await validatePath(parsed.data.path, allowedDirectories, logger, config);
            const stats = await getFileStats(validPath, logger, config);
            return { content: [{ type: 'text', text: JSON.stringify(stats, null, 2) }] };
        }

        case 'list_allowed_directories': {
            return { content: [{ type: 'text', text: JSON.stringify(allowedDirectories, null, 2) }] };
        }

        case 'server_stats': {
            const stats = { requestCount, editOperationCount, config };
            return { content: [{ type: 'text', text: JSON.stringify(stats, null, 2) }] };
        }

        default:
          throw createError('UNKNOWN_TOOL', `Unknown tool: ${name}`);
      }
    } catch (error) {
      let structuredError: StructuredError;
      if ((error as any).code) {
        structuredError = error as StructuredError;
      } else {
        structuredError = createError('UNKNOWN_ERROR', error instanceof Error ? error.message : String(error));
      }
      logger.error({ error: structuredError, tool: name }, `Tool request failed: ${name}`);
      return {
        content: [{ type: 'text', text: `Error (${structuredError.code}): ${structuredError.message}` }],
        isError: true,
        meta: { hint: structuredError.hint, confidence: structuredError.confidence, details: structuredError.details }
      };
    }
  });
}
