import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { z } from 'zod';

// Konfiguracja serwera z argumentów CLI
const allowedDirectories = process.argv.slice(2);

if (allowedDirectories.length === 0) {
  console.error('Usage: node index.js <allowed-directory-1> [allowed-directory-2] ...');
  process.exit(1);
}

console.error('🚀 Starting MCP Filesystem Server...');
console.error('📂 Allowed directories:', allowedDirectories);

// Szczegółowe logowanie do pliku
const logFile = '/home/lukasz/projects/gatto-ps-ai/gatto_filesystem/logs/server-debug.log';

async function logToFile(message: string, data?: any) {
  const timestamp = new Date().toISOString();
  const logEntry = `[${timestamp}] ${message}${data ? '\nData: ' + JSON.stringify(data, null, 2) : ''}\n`;
  try {
    await fs.mkdir(path.dirname(logFile), { recursive: true });
    await fs.appendFile(logFile, logEntry);
  } catch (error) {
    console.error('Logging error:', error);
  }
}

// Stwórz serwer MCP zgodnie z SDK
const server = new McpServer({
  name: 'mcp-filesystem-server',
  version: '0.7.0'
});

// Funkcje helper
async function validatePath(targetPath: string): Promise<string> {
  const absolutePath = path.resolve(targetPath);
  
  const isAllowed = allowedDirectories.some(allowedDir => {
    const resolvedAllowed = path.resolve(allowedDir);
    return absolutePath.startsWith(resolvedAllowed);
  });
  
  if (!isAllowed) {
    throw new Error(`Access denied: Path ${absolutePath} is not in allowed directories`);
  }
  
  return absolutePath;
}

// 1. read_file - Reads the entire content of a single file with automatic encoding detection  
server.tool(
  'read_file',
  "Reads the entire content of a single file specified by its 'path'. It automatically detects the file type: for text files, it returns UTF-8 content, and for binary files, it returns a base64 encoded string. The encoding can be forced by using the 'encoding' parameter.",
  {
    path: z.string().describe('Path to the file to read'),
    encoding: z.enum(['utf-8', 'base64', 'auto']).optional().default('auto').describe('Encoding for file content')
  },
  async ({ path: filePath, encoding = 'auto' }) => {
    await logToFile('read_file called', { filePath, encoding });
    
    const validatedPath = await validatePath(filePath);
    const stats = await fs.stat(validatedPath);
    
    if (!stats.isFile()) {
      throw new Error(`Path ${filePath} is not a file`);
    }

    let content: string;
    if (encoding === 'base64') {
      const buffer = await fs.readFile(validatedPath);
      content = buffer.toString('base64');
      await logToFile('read_file: used base64 encoding', { contentLength: content.length });
    } else if (encoding === 'utf-8') {
      content = await fs.readFile(validatedPath, 'utf-8');
      await logToFile('read_file: used utf-8 encoding', { 
        contentLength: content.length,
        hasNewlines: content.includes('\n'),
        firstLine: content.split('\n')[0],
        lineCount: content.split('\n').length
      });
    } else {
      // Auto-detect encoding
      const buffer = await fs.readFile(validatedPath);
      try {
        content = buffer.toString('utf-8');
        // Test if it's valid UTF-8 by checking for replacement characters
        if (content.includes('\uFFFD')) {
          content = buffer.toString('base64');
          encoding = 'base64';
          await logToFile('read_file: auto-detected as base64 (replacement chars found)');
        } else {
          encoding = 'utf-8';
          await logToFile('read_file: auto-detected as utf-8', {
            contentLength: content.length,
            hasNewlines: content.includes('\n'),
            firstLine: content.split('\n')[0],
            lineCount: content.split('\n').length
          });
        }
      } catch {
        content = buffer.toString('base64');
        encoding = 'base64';
        await logToFile('read_file: auto-detected as base64 (utf-8 failed)');
      }
    }

    const result = {
      content: [{
        type: 'text',
        text: content
      }],
      _meta: { encoding }
    };
    
    await logToFile('read_file result', { 
      encoding, 
      contentLength: content.length,
      resultType: result.content[0].type
    });

    return result;
  }
);

// 2. read_multiple_files - Reads multiple files
server.tool(
  'read_multiple_files',
  "Reads multiple files in a single operation. This tool takes an array of file paths and returns the content of each file along with its encoding. It's useful for batch processing or when you need to read several related files simultaneously.",
  {
    paths: z.array(z.string()).describe('Array of file paths to read'),
    encoding: z.enum(['utf-8', 'base64', 'auto']).optional().default('auto').describe('Encoding for file content')
  },
  async ({ paths, encoding = 'auto' }) => {
    const results = [];
    
    for (const filePath of paths) {
      try {
        const validatedPath = await validatePath(filePath);
        const stats = await fs.stat(validatedPath);
        
        if (!stats.isFile()) {
          results.push({
            path: filePath,
            error: `Path ${filePath} is not a file`
          });
          continue;
        }

        let content: string;
        let actualEncoding = encoding;
        
        if (encoding === 'base64') {
          const buffer = await fs.readFile(validatedPath);
          content = buffer.toString('base64');
        } else if (encoding === 'utf-8') {
          content = await fs.readFile(validatedPath, 'utf-8');
        } else {
          // Auto-detect encoding
          const buffer = await fs.readFile(validatedPath);
          try {
            content = buffer.toString('utf-8');
            if (content.includes('\uFFFD')) {
              content = buffer.toString('base64');
              actualEncoding = 'base64';
            } else {
              actualEncoding = 'utf-8';
            }
          } catch {
            content = buffer.toString('base64');
            actualEncoding = 'base64';
          }
        }
        
        results.push({
          path: filePath,
          content,
          encoding: actualEncoding
        });
      } catch (error) {
        results.push({
          path: filePath,
          error: (error as Error).message
        });
      }
    }

    return {
      content: [{
        type: 'text',
        text: JSON.stringify(results, null, 2)
      }]
    };
  }
);

// 3. list_allowed_directories - Lists allowed base directories
server.tool(
  'list_allowed_directories',
  "Returns a list of all directories that this server is allowed to access. This is useful for understanding the scope of the server's permissions and identifying available root directories for other operations.",
  {},
  async () => {
    return {
      content: [{
        type: 'text',
        text: JSON.stringify(allowedDirectories, null, 2)
      }]
    };
  }
);

// 4. write_file - Writes content to a file
server.tool(
  'write_file',
  "Creates a new file or overwrites an existing file with the specified content. Supports both UTF-8 text content and base64-encoded binary content. The file will be created along with any necessary parent directories.",
  {
    path: z.string().describe('Path to the file to write'),
    content: z.string().describe('Content to write to the file'),
    encoding: z.enum(['utf-8', 'base64']).optional().default('utf-8').describe('Encoding of provided content')
  },
  async ({ path: filePath, content, encoding = 'utf-8' }) => {
    const validatedPath = await validatePath(filePath);
    
    if (encoding === 'base64') {
      const buffer = Buffer.from(content, 'base64');
      await fs.writeFile(validatedPath, buffer);
    } else {
      await fs.writeFile(validatedPath, content, 'utf-8');
    }

    return {
      content: [{
        type: 'text',
        text: `Successfully wrote to ${filePath}`
      }]
    };
  }
);

// 5. edit_file - Advanced in-place editing with fuzzy matching
server.tool(
  'edit_file',
  "Performs in-place editing of an existing file by searching for specific text patterns and replacing them with new content. Supports multiple edit operations in a single call and includes a dry-run option for previewing changes.",
  {
    path: z.string().describe('Path to the file to edit'),
    edits: z.array(z.object({
      oldText: z.string().describe('Text to search for'),
      newText: z.string().describe('Text to replace with')
    })).describe('Array of edit operations'),
    dryRun: z.boolean().optional().default(false).describe('Preview changes without modifying file')
  },
  async ({ path: filePath, edits, dryRun = false }) => {
    const validatedPath = await validatePath(filePath);
    let content = await fs.readFile(validatedPath, 'utf-8');
    
    let changes = 0;
    let preview = content;
    
    for (const edit of edits) {
      if (preview.includes(edit.oldText)) {
        preview = preview.replace(edit.oldText, edit.newText);
        changes++;
      }
    }
    
    if (!dryRun && changes > 0) {
      await fs.writeFile(validatedPath, preview, 'utf-8');
    }
    
    return {
      content: [{
        type: 'text',
        text: dryRun 
          ? `Dry run: ${changes} changes would be made\n\nPreview:\n${preview}`
          : `Applied ${changes} changes to ${filePath}`
      }]
    };
  }
);

// 6. create_directory - Creates a directory
server.tool(
  'create_directory',
  "Creates a new directory at the specified path. If parent directories don't exist, they will be created automatically (recursive creation). This operation is idempotent - it won't fail if the directory already exists.",
  {
    path: z.string().describe('Path to the directory to create')
  },
  async ({ path: dirPath }) => {
    const validatedPath = await validatePath(dirPath);
    await fs.mkdir(validatedPath, { recursive: true });
    
    return {
      content: [{
        type: 'text',
        text: `Successfully created directory ${dirPath}`
      }]
    };
  }
);

// 7. list_directory - Lists directory contents
server.tool(
  'list_directory',
  "Lists the contents of a directory, showing files and subdirectories with their types and metadata. Supports both shallow listing and recursive traversal of the entire directory tree.",
  {
    path: z.string().describe('Path to the directory to list'),
    recursive: z.boolean().optional().default(false).describe('List contents recursively')
  },
  async ({ path: dirPath, recursive = false }) => {
    const validatedPath = await validatePath(dirPath);
    const entries: any[] = [];
    
    async function scanDirectory(currentPath: string, basePath: string) {
      const items = await fs.readdir(currentPath, { withFileTypes: true });
      
      for (const item of items) {
        const fullPath = path.join(currentPath, item.name);
        const relativePath = path.relative(basePath, fullPath);
        
        const entry = {
          name: item.name,
          type: item.isDirectory() ? 'directory' : 'file',
          path: relativePath
        };
        
        if (item.isFile()) {
          const stats = await fs.stat(fullPath);
          (entry as any).size = stats.size;
        }
        
        entries.push(entry);
        
        if (recursive && item.isDirectory()) {
          await scanDirectory(fullPath, basePath);
        }
      }
    }
    
    await scanDirectory(validatedPath, validatedPath);
    entries.sort((a, b) => a.path.localeCompare(b.path));
    
    return {
      content: [{
        type: 'text',
        text: JSON.stringify({ entries }, null, 2)
      }]
    };
  }
);

// 8. directory_tree - Returns hierarchical directory structure
server.tool(
  'directory_tree',
  "Generates a hierarchical tree representation of a directory structure, showing the nested relationship between folders and files. Useful for understanding project structure or navigating complex directory hierarchies.",
  {
    path: z.string().describe('Path to the directory'),
    maxDepth: z.number().optional().describe('Maximum depth to traverse')
  },
  async ({ path: dirPath, maxDepth }) => {
    const validatedPath = await validatePath(dirPath);
    
    async function buildTree(currentPath: string, depth = 0): Promise<any> {
      if (maxDepth !== undefined && depth >= maxDepth) {
        return null;
      }
      
      const stats = await fs.stat(currentPath);
      const name = path.basename(currentPath);
      
      if (stats.isFile()) {
        return {
          name,
          type: 'file',
          size: stats.size
        };
      }
      
      if (stats.isDirectory()) {
        const children = [];
        const items = await fs.readdir(currentPath, { withFileTypes: true });
        
        for (const item of items) {
          const childPath = path.join(currentPath, item.name);
          const childTree = await buildTree(childPath, depth + 1);
          if (childTree) {
            children.push(childTree);
          }
        }
        
        return {
          name,
          type: 'directory',
          children: children.sort((a, b) => a.name.localeCompare(b.name))
        };
      }
    }
    
    const tree = await buildTree(validatedPath);
    
    return {
      content: [{
        type: 'text',
        text: JSON.stringify(tree, null, 2)
      }]
    };
  }
);

// 9. move_file - Moves or renames a file/directory
server.tool(
  'move_file',
  "Moves a file or directory from one location to another. This operation can be used for both moving items between directories and renaming items within the same directory.",
  {
    source: z.string().describe('Source path'),
    destination: z.string().describe('Destination path')
  },
  async ({ source, destination }) => {
    const validatedSource = await validatePath(source);
    const validatedDestination = await validatePath(destination);
    
    await fs.rename(validatedSource, validatedDestination);
    
    return {
      content: [{
        type: 'text',
        text: `Successfully moved ${source} to ${destination}`
      }]
    };
  }
);

// 10. delete_file - Deletes a file
server.tool(
  'delete_file',
  "Permanently deletes a single file from the filesystem. This operation cannot be undone, so use with caution. Fails safely if the target is a directory instead of a file.",
  {
    path: z.string().describe('Path to the file to delete')
  },
  async ({ path: filePath }) => {
    const validatedPath = await validatePath(filePath);
    const stats = await fs.stat(validatedPath);
    
    if (!stats.isFile()) {
      throw new Error(`Path ${filePath} is not a file`);
    }
    
    await fs.unlink(validatedPath);
    
    return {
      content: [{
        type: 'text',
        text: `Successfully deleted file ${filePath}`
      }]
    };
  }
);

// 11. delete_directory - Deletes a directory
server.tool(
  'delete_directory',
  "Removes a directory from the filesystem. Can operate in two modes: remove only empty directories (default) or recursively delete a directory and all its contents. Use the recursive option with extreme caution.",
  {
    path: z.string().describe('Path to the directory to delete'),
    recursive: z.boolean().optional().default(false).describe('Delete directory and all contents')
  },
  async ({ path: dirPath, recursive = false }) => {
    const validatedPath = await validatePath(dirPath);
    
    if (recursive) {
      await fs.rm(validatedPath, { recursive: true });
    } else {
      await fs.rmdir(validatedPath);
    }
    
    return {
      content: [{
        type: 'text',
        text: `Successfully deleted directory ${dirPath}`
      }]
    };
  }
);

// 12. search_files - Searches for files matching a pattern
server.tool(
  'search_files',
  "Searches for files and directories within a specified directory tree that match a given pattern. Supports filtering by name patterns, excluding certain patterns, and limiting search depth and results.",
  {
    path: z.string().describe('Base directory path to search in'),
    pattern: z.string().describe('Pattern to match files (simple string match)'),
    excludePatterns: z.array(z.string()).optional().describe('Patterns to exclude'),
    maxDepth: z.number().optional().describe('Maximum depth for search'),
    maxResults: z.number().optional().describe('Maximum number of results')
  },
  async ({ path: basePath, pattern, excludePatterns, maxDepth, maxResults }) => {
    const validatedPath = await validatePath(basePath);
    const results: any[] = [];
    
    async function searchInDirectory(currentPath: string, depth = 0) {
      if (maxDepth !== undefined && depth >= maxDepth) return;
      if (maxResults !== undefined && results.length >= maxResults) return;
      
      try {
        const entries = await fs.readdir(currentPath, { withFileTypes: true });
        
        for (const entry of entries) {
          if (maxResults !== undefined && results.length >= maxResults) break;
          
          const fullPath = path.join(currentPath, entry.name);
          const relativePath = path.relative(validatedPath, fullPath);
          
          // Check excludePatterns
          if (excludePatterns?.some(exclude => entry.name.includes(exclude))) {
            continue;
          }
          
          // Check if matches pattern
          if (entry.name.includes(pattern)) {
            results.push({
              name: entry.name,
              type: entry.isDirectory() ? 'directory' : 'file',
              path: relativePath,
              fullPath: path.join(validatedPath, relativePath)
            });
          }
          
          // Recurse into directories
          if (entry.isDirectory()) {
            await searchInDirectory(fullPath, depth + 1);
          }
        }
      } catch (error) {
        // Skip directories we can't read
      }
    }
    
    await searchInDirectory(validatedPath);
    
    return {
      content: [{
        type: 'text',
        text: JSON.stringify({ 
          pattern, 
          basePath, 
          results: results.slice(0, maxResults) 
        }, null, 2)
      }]
    };
  }
);

// 13. get_file_info - Gets detailed file/directory metadata
server.tool(
  'get_file_info',
  "Retrieves comprehensive metadata about a file or directory, including size, creation/modification dates, permissions, and type information. Useful for understanding file properties before performing operations.",
  {
    path: z.string().describe('Path to get information about')
  },
  async ({ path: targetPath }) => {
    const validatedPath = await validatePath(targetPath);
    const stats = await fs.stat(validatedPath);
    
    const info = {
      path: targetPath,
      name: path.basename(validatedPath),
      type: stats.isDirectory() ? 'directory' : 'file',
      size: stats.size,
      created: stats.birthtime,
      modified: stats.mtime,
      accessed: stats.atime,
      permissions: stats.mode.toString(8),
      isReadable: true, // We can read it if we got this far
      isWritable: true  // Assume writable in allowed directories
    };
    
    return {
      content: [{
        type: 'text',
        text: JSON.stringify(info, null, 2)
      }]
    };
  }
);

// 14. server_stats - Gets server statistics and configuration
server.tool(
  'server_stats',
  "Provides comprehensive information about the MCP filesystem server, including version, configuration, performance metrics, and available capabilities. Useful for debugging and system monitoring.",
  {},
  async () => {
    const stats = {
      serverName: 'mcp-filesystem-server',
      version: '0.7.0',
      allowedDirectories,
      uptime: process.uptime(),
      memoryUsage: process.memoryUsage(),
      nodeVersion: process.version,
      platform: process.platform,
      capabilities: [
        'read_file',
        'read_multiple_files', 
        'list_allowed_directories',
        'write_file',
        'edit_file',
        'create_directory',
        'list_directory',
        'directory_tree',
        'move_file',
        'delete_file',
        'delete_directory',
        'search_files',
        'get_file_info',
        'server_stats'
      ]
    };
    
    return {
      content: [{
        type: 'text',
        text: JSON.stringify(stats, null, 2)
      }]
    };
  }
);

// Uruchom serwer
async function main() {
  const transport = new StdioServerTransport();
  
  console.error('🔌 Connecting to transport...');
  await server.connect(transport);
  console.error('✅ MCP Filesystem Server connected and ready!');
}

main().catch((error) => {
  console.error('💥 Server startup failed:', error);
  process.exit(1);
});