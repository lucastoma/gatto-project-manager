import { z } from "zod";
import fs from "node:fs/promises";
import path from "node:path";

export const ConfigSchema = z.object({
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

export type Config = z.infer<typeof ConfigSchema>;

export async function loadConfig(): Promise<Config> {
    const args = process.argv.slice(2);
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
            return ConfigSchema.parse(rawConfig);
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
        
        const DEFAULT_MAX_DISTANCE_RATIO = parseFloat(process.env.MCP_EDIT_MAX_DISTANCE_RATIO || '0.25');
        const DEFAULT_MIN_SIMILARITY = parseFloat(process.env.MCP_EDIT_MIN_SIMILARITY || '0.7');
        const DEFAULT_CASE_SENSITIVE = process.env.MCP_EDIT_CASE_SENSITIVE === 'true';
        const DEFAULT_IGNORE_WHITESPACE = process.env.MCP_EDIT_IGNORE_WHITESPACE !== 'false';

        return {
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
}
