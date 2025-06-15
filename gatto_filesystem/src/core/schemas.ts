import { z } from 'zod';
import type { Config } from '../server/config.js';

export const HandshakeRequestSchema = z.object({
  method: z.literal('handshake'),
  params: z.object({}).optional()
});

export const ListToolsRequestSchema = z.object({
  method: z.literal('list_tools'),
  params: z.object({}).optional()
});

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

export const EditOperationSchema = z.object({
  oldText: z.string().describe('Text to search for - can be slightly inaccurate'),
  newText: z.string().describe('Text to replace with'),
  forcePartialMatch: z.boolean().optional().default(false)
    .describe('If true, allows partial matches above minSimilarity threshold when no exact match is found')
});
export type EditOperation = z.infer<typeof EditOperationSchema>;

export const getEditFileArgsSchema = (config: Config) => z.object({
  path: z.string(),
  edits: z.array(EditOperationSchema),
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

export const ListDirectoryEntrySchema = z.object({
  name: z.string().describe('Name of the file or directory'),
  path: z.string().describe('Relative path from the base allowed directory'),
  type: z.enum(['file', 'directory', 'symlink', 'other']).describe('Type of the entry'),
  size: z.number().optional().describe('Size of the file in bytes, undefined for directories or if error reading stats')
});
export type ListDirectoryEntry = z.infer<typeof ListDirectoryEntrySchema>;

export const ListDirectoryArgsSchema = z.object({
  path: z.string(),
});

// Zaktualizowany schemat z dodanym `maxDepth`
export const DirectoryTreeArgsSchema = z.object({
  path: z.string(),
  maxDepth: z.number().int().positive().optional().describe('Maximum depth to traverse the directory tree')
});

// Define the recursive DirectoryTreeEntrySchema
// We need to use z.lazy to handle recursive types with Zod
export const DirectoryTreeEntrySchema: z.ZodType<DirectoryTreeEntry> = z.lazy(() =>
  z.object({
    name: z.string().describe('Name of the file or directory'),
    path: z.string().describe('Full absolute path of the file or directory'),
    type: z.enum(['file', 'directory']).describe('Type of the entry'),
    children: z.array(DirectoryTreeEntrySchema).optional().describe('Children of the directory entry, undefined for files'),
  })
);

// Define the TypeScript interface for DirectoryTreeEntry for clarity
export interface DirectoryTreeEntry {
  name: string;
  path: string;
  type: 'file' | 'directory';
  children?: DirectoryTreeEntry[];
}

// The result schema is essentially the root entry of the directory tree
export const DirectoryTreeResultSchema = DirectoryTreeEntrySchema;

export const MoveFileArgsSchema = z.object({
  source: z.string(),
  destination: z.string(),
});

export const ListAllowedDirectoriesArgsSchema = z.object({}); // No parameters for listing allowed directories

export const ServerStatsArgsSchema = z.object({}); // Schema for server_stats tool arguments

export const SearchFilesArgsSchema = z.object({
  path: z.string(),
  pattern: z.string(),
  excludePatterns: z.array(z.string()).optional().default([]),
  useExactPatterns: z.boolean().default(false).describe('Use patterns exactly as provided instead of wrapping with **/'),
  maxDepth: z.number().int().positive().optional().describe('Maximum depth to search'),
  maxResults: z.number().int().positive().optional().describe('Maximum number of results to return')
});

export const GetFileInfoArgsSchema = z.object({
  path: z.string(),
});

export const CallToolRequestSchema = z.object({
  method: z.literal('call_tool'),
  params: z.object({
    name: z.string(),
    args: z.any()
  })
});

export const DeleteFileArgsSchema = z.object({
  path: z.string(),
});

export const DeleteDirectoryArgsSchema = z.object({
  path: z.string(),
  recursive: z.boolean().optional().default(false).describe('Recursively delete directory contents')
});