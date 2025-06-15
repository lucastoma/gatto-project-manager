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

// Tool: list_directory
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

// Tool: read_file
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

// Tool: write_file  
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