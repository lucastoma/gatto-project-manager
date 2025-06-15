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

// ===== WSZYSTKIE 14 TOOLS =====

// 1. read_file
server.tool(
  'read_file',
  {
    path: z.string().describe('File path to read'),
    encoding: z.string().optional().default('utf-8').describe('File encoding')
  },
  async ({ path: filePath, encoding }: { path: string; encoding: string }) => {
    console.error('🔧 Executing read_file:', { filePath, encoding });
    
    try {
      const validatedPath = await validatePath(filePath);
      const content = await fs.readFile(validatedPath, encoding as BufferEncoding);
      
      const result = {
        content,
        encoding
      };
      
      console.error('✅ read_file success');
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(result, null, 2)
          }
        ]
      };
    } catch (error) {
      console.error('❌ read_file error:', error);
      return {
        content: [
          {
            type: 'text',
            text: `Error: ${error instanceof Error ? error.message : String(error)}`
          }
        ],
        isError: true
      };
    }
  }
);

// 2. read_multiple_files
server.tool(
  'read_multiple_files',
  {
    paths: z.array(z.string()).describe('Array of file paths to read')
  },
  async ({ paths }: { paths: string[] }) => {
    console.error('🔧 Executing read_multiple_files:', { paths });
    
    const results = await Promise.allSettled(
      paths.map(async (filePath) => {
        try {
          const validatedPath = await validatePath(filePath);
          const content = await fs.readFile(validatedPath, 'utf-8');
          return { path: filePath, content, success: true };
        } catch (error) {
          return { 
            path: filePath, 
            error: error instanceof Error ? error.message : String(error),
            success: false 
          };
        }
      })
    );
    
    console.error('✅ read_multiple_files success');
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(results.map(r => r.status === 'fulfilled' ? r.value : r.reason), null, 2)
        }
      ]
    };
  }
);

// 3. list_allowed_directories
server.tool(
  'list_allowed_directories',
  {},
  async () => {
    console.error('🔧 Executing list_allowed_directories');
    
    const result = {
      allowedDirectories: allowedDirectories
    };
    
    console.error('✅ list_allowed_directories success');
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(result, null, 2)
        }
      ]
    };
  }
);

// 4. write_file
server.tool(
  'write_file',
  {
    path: z.string().describe('File path to write'),
    content: z.string().describe('Content to write'),
    encoding: z.string().optional().default('utf-8').describe('File encoding')
  },
  async ({ path: filePath, content, encoding }: { path: string; content: string; encoding: string }) => {
    console.error('🔧 Executing write_file:', { filePath, encoding });
    
    try {
      const validatedPath = await validatePath(filePath);
      await fs.writeFile(validatedPath, content, encoding as BufferEncoding);
      
      console.error('✅ write_file success');
      return {
        content: [
          {
            type: 'text',
            text: `File written successfully to ${filePath}`
          }
        ]
      };
    } catch (error) {
      console.error('❌ write_file error:', error);
      return {
        content: [
          {
            type: 'text',
            text: `Error: ${error instanceof Error ? error.message : String(error)}`
          }
        ],
        isError: true
      };
    }
  }
);

// 5. edit_file
server.tool(
  'edit_file',
  {
    path: z.string().describe('File path to edit'),
    oldText: z.string().describe('Text to find and replace'),
    newText: z.string().describe('New text to replace with'),
    dryRun: z.boolean().optional().default(false).describe('Preview changes without modifying file')
  },
  async ({ path: filePath, oldText, newText, dryRun }: { path: string; oldText: string; newText: string; dryRun: boolean }) => {
    console.error('🔧 Executing edit_file:', { filePath, dryRun });
    
    try {
      const validatedPath = await validatePath(filePath);
      const content = await fs.readFile(validatedPath, 'utf-8');
      
      if (!content.includes(oldText)) {
        throw new Error(`Text not found: ${oldText}`);
      }
      
      const newContent = content.replace(oldText, newText);
      
      if (dryRun) {
        const result = {
          preview: true,
          changes: {
            oldText,
            newText,
            occurrences: (content.match(new RegExp(oldText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')) || []).length
          }
        };
        
        console.error('✅ edit_file (dry run) success');
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2)
            }
          ]
        };
      } else {
        await fs.writeFile(validatedPath, newContent, 'utf-8');
        
        console.error('✅ edit_file success');
        return {
          content: [
            {
              type: 'text',
              text: `File edited successfully: ${filePath}`
            }
          ]
        };
      }
    } catch (error) {
      console.error('❌ edit_file error:', error);
      return {
        content: [
          {
            type: 'text',
            text: `Error: ${error instanceof Error ? error.message : String(error)}`
          }
        ],
        isError: true
      };
    }
  }
);

// 6. create_directory
server.tool(
  'create_directory',
  {
    path: z.string().describe('Directory path to create')
  },
  async ({ path: dirPath }: { path: string }) => {
    console.error('🔧 Executing create_directory:', { dirPath });
    
    try {
      const validatedPath = await validatePath(dirPath);
      await fs.mkdir(validatedPath, { recursive: true });
      
      console.error('✅ create_directory success');
      return {
        content: [
          {
            type: 'text',
            text: `Directory created successfully: ${dirPath}`
          }
        ]
      };
    } catch (error) {
      console.error('❌ create_directory error:', error);
      return {
        content: [
          {
            type: 'text',
            text: `Error: ${error instanceof Error ? error.message : String(error)}`
          }
        ],
        isError: true
      };
    }
  }
);

// 7. list_directory
server.tool(
  'list_directory',
  {
    path: z.string().describe('Directory path to list'),
    recursive: z.boolean().optional().default(false).describe('List recursively')
  },
  async ({ path: dirPath, recursive }: { path: string; recursive: boolean }) => {
    console.error('🔧 Executing list_directory:', { dirPath, recursive });
    
    try {
      const validatedPath = await validatePath(dirPath);
      const entries = await fs.readdir(validatedPath, { withFileTypes: true });
      
      const result = {
        entries: entries.map(entry => ({
          name: entry.name,
          type: entry.isDirectory() ? 'directory' : 'file',
          path: entry.name
        }))
      };
      
      console.error('✅ list_directory success');
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(result, null, 2)
          }
        ]
      };
    } catch (error) {
      console.error('❌ list_directory error:', error);
      return {
        content: [
          {
            type: 'text',
            text: `Error: ${error instanceof Error ? error.message : String(error)}`
          }
        ],
        isError: true
      };
    }
  }
);

// 8. directory_tree
server.tool(
  'directory_tree',
  {
    path: z.string().describe('Directory path to get tree structure'),
    maxDepth: z.number().optional().describe('Maximum depth for traversal')
  },
  async ({ path: dirPath, maxDepth }: { path: string; maxDepth?: number }) => {
    console.error('🔧 Executing directory_tree:', { dirPath, maxDepth });
    
    try {
      const validatedPath = await validatePath(dirPath);
      
      async function buildTree(currentPath: string, currentDepth: number = 0): Promise<any> {
        if (maxDepth !== undefined && currentDepth >= maxDepth) {
          return null;
        }
        
        const stats = await fs.stat(currentPath);
        const name = path.basename(currentPath);
        
        if (stats.isDirectory()) {
          const entries = await fs.readdir(currentPath);
          const children = await Promise.all(
            entries.map(async (entry) => {
              const entryPath = path.join(currentPath, entry);
              return buildTree(entryPath, currentDepth + 1);
            })
          );
          
          return {
            name,
            type: 'directory',
            children: children.filter(Boolean)
          };
        } else {
          return {
            name,
            type: 'file'
          };
        }
      }
      
      const tree = await buildTree(validatedPath);
      
      console.error('✅ directory_tree success');
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(tree, null, 2)
          }
        ]
      };
    } catch (error) {
      console.error('❌ directory_tree error:', error);
      return {
        content: [
          {
            type: 'text',
            text: `Error: ${error instanceof Error ? error.message : String(error)}`
          }
        ],
        isError: true
      };
    }
  }
);

// 9. move_file
server.tool(
  'move_file',
  {
    source: z.string().describe('Source file path'),
    destination: z.string().describe('Destination file path')
  },
  async ({ source, destination }: { source: string; destination: string }) => {
    console.error('🔧 Executing move_file:', { source, destination });
    
    try {
      const validatedSource = await validatePath(source);
      const validatedDestination = await validatePath(destination);
      await fs.rename(validatedSource, validatedDestination);
      
      console.error('✅ move_file success');
      return {
        content: [
          {
            type: 'text',
            text: `File moved successfully from ${source} to ${destination}`
          }
        ]
      };
    } catch (error) {
      console.error('❌ move_file error:', error);
      return {
        content: [
          {
            type: 'text',
            text: `Error: ${error instanceof Error ? error.message : String(error)}`
          }
        ],
        isError: true
      };
    }
  }
);

// 10. delete_file
server.tool(
  'delete_file',
  {
    path: z.string().describe('File path to delete')
  },
  async ({ path: filePath }: { path: string }) => {
    console.error('🔧 Executing delete_file:', { filePath });
    
    try {
      const validatedPath = await validatePath(filePath);
      await fs.unlink(validatedPath);
      
      console.error('✅ delete_file success');
      return {
        content: [
          {
            type: 'text',
            text: `File deleted successfully: ${filePath}`
          }
        ]
      };
    } catch (error) {
      console.error('❌ delete_file error:', error);
      return {
        content: [
          {
            type: 'text',
            text: `Error: ${error instanceof Error ? error.message : String(error)}`
          }
        ],
        isError: true
      };
    }
  }
);

// 11. delete_directory
server.tool(
  'delete_directory',
  {
    path: z.string().describe('Directory path to delete'),
    recursive: z.boolean().optional().default(false).describe('Delete recursively')
  },
  async ({ path: dirPath, recursive }: { path: string; recursive: boolean }) => {
    console.error('🔧 Executing delete_directory:', { dirPath, recursive });
    
    try {
      const validatedPath = await validatePath(dirPath);
      if (recursive) {
        await fs.rm(validatedPath, { recursive: true });
      } else {
        await fs.rmdir(validatedPath);
      }
      
      console.error('✅ delete_directory success');
      return {
        content: [
          {
            type: 'text',
            text: `Directory deleted successfully: ${dirPath}`
          }
        ]
      };
    } catch (error) {
      console.error('❌ delete_directory error:', error);
      return {
        content: [
          {
            type: 'text',
            text: `Error: ${error instanceof Error ? error.message : String(error)}`
          }
        ],
        isError: true
      };
    }
  }
);

// 12. search_files
server.tool(
  'search_files',
  {
    path: z.string().describe('Base directory path to search in'),
    pattern: z.string().describe('Pattern to match files (simple string match)'),
    maxDepth: z.number().optional().describe('Maximum depth for search')
  },
  async ({ path: basePath, pattern, maxDepth }: { path: string; pattern: string; maxDepth?: number }) => {
    console.error('🔧 Executing search_files:', { basePath, pattern });
    
    try {
      const validatedPath = await validatePath(basePath);
      
      async function searchFiles(dir: string, searchPattern: string, currentDepth: number = 0): Promise<string[]> {
        if (maxDepth !== undefined && currentDepth >= maxDepth) {
          return [];
        }
        
        const entries = await fs.readdir(dir, { withFileTypes: true });
        const results: string[] = [];
        
        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          const relativePath = path.relative(validatedPath, fullPath);
          
          if (entry.isFile() && entry.name.includes(searchPattern)) {
            results.push(relativePath);
          } else if (entry.isDirectory()) {
            const subResults = await searchFiles(fullPath, searchPattern, currentDepth + 1);
            results.push(...subResults);
          }
        }
        
        return results;
      }
      
      const matches = await searchFiles(validatedPath, pattern);
      
      const results = matches.map(match => ({
        path: match,
        fullPath: path.join(validatedPath, match)
      }));
      
      console.error('✅ search_files success');
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(results, null, 2)
          }
        ]
      };
    } catch (error) {
      console.error('❌ search_files error:', error);
      return {
        content: [
          {
            type: 'text',
            text: `Error: ${error instanceof Error ? error.message : String(error)}`
          }
        ],
        isError: true
      };
    }
  }
);

// 13. get_file_info
server.tool(
  'get_file_info',
  {
    path: z.string().describe('File or directory path to get info for')
  },
  async ({ path: filePath }: { path: string }) => {
    console.error('🔧 Executing get_file_info:', { filePath });
    
    try {
      const validatedPath = await validatePath(filePath);
      const stats = await fs.stat(validatedPath);
      
      const result = {
        path: filePath,
        size: stats.size,
        isFile: stats.isFile(),
        isDirectory: stats.isDirectory(),
        created: stats.birthtime,
        modified: stats.mtime,
        accessed: stats.atime,
        permissions: stats.mode
      };
      
      console.error('✅ get_file_info success');
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(result, null, 2)
          }
        ]
      };
    } catch (error) {
      console.error('❌ get_file_info error:', error);
      return {
        content: [
          {
            type: 'text',
            text: `Error: ${error instanceof Error ? error.message : String(error)}`
          }
        ],
        isError: true
      };
    }
  }
);

// 14. server_stats
server.tool(
  'server_stats',
  {},
  async () => {
    console.error('🔧 Executing server_stats');
    
    const stats = {
      serverInfo: {
        name: 'mcp-filesystem-server',
        version: '0.7.0'
      },
      configuration: {
        allowedDirectories: allowedDirectories
      },
      runtime: {
        nodeVersion: process.version,
        platform: process.platform,
        architecture: process.arch,
        uptime: process.uptime()
      }
    };
    
    console.error('✅ server_stats success');
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(stats, null, 2)
        }
      ]
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