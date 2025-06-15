import { z } from "zod";
import fs from "node:fs/promises";
import path from "node:path";
import { DEFAULT_EXCLUDE_PATTERNS, DEFAULT_ALLOWED_EXTENSIONS } from '../constants/extensions';

export const ConfigSchema = z.object({
  allowedDirectories: z.array(z.string()),
  fileFiltering: z.object({
    defaultExcludes: z.array(z.string()).default(DEFAULT_EXCLUDE_PATTERNS),
    allowedExtensions: z.array(z.string()).default(DEFAULT_ALLOWED_EXTENSIONS),
    forceTextFiles: z.boolean().default(true)
  }).default({
    defaultExcludes: DEFAULT_EXCLUDE_PATTERNS,
    allowedExtensions: DEFAULT_ALLOWED_EXTENSIONS,
    forceTextFiles: true
  }),
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
    maxConcurrentEdits: z.number().positive().default(10),
    maxGlobalConcurrentEdits: z.number().positive().default(20)
  }).default({}),
  limits: z.object({
    maxReadBytes: z.number().positive().default(5 * 1024 * 1024), // 5 MB
    maxWriteBytes: z.number().positive().default(5 * 1024 * 1024) // 5 MB
  }).default({})
}).default({
  allowedDirectories: [],
  fileFiltering: {
    defaultExcludes: DEFAULT_EXCLUDE_PATTERNS,
    allowedExtensions: DEFAULT_ALLOWED_EXTENSIONS,
    forceTextFiles: true
  },
  concurrency: {
    maxConcurrentEdits: 10,
    maxGlobalConcurrentEdits: 20
  },
  limits: {
    maxReadBytes: 5 * 1024 * 1024,
    maxWriteBytes: 5 * 1024 * 1024
  }
});

export type Config = z.infer<typeof ConfigSchema>;

export async function getConfig(args: string[]): Promise<Config> {
  if (args.length > 0 && args[0] === '--config') {
    if (args.length < 2) {
      console.error("Error: --config requires a path to a config file.");
      process.exit(1);
    }
    const configFile = args[1];
    try {
      const configContent = await fs.readFile(configFile, 'utf-8');
      return ConfigSchema.parse(JSON.parse(configContent));
    } catch (err) {
      console.error(`Error loading config file ${configFile}:`, err);
      process.exit(1);
    }
  }

  // Default config with file filtering
  return ConfigSchema.parse({
    allowedDirectories: args.length > 0 ? args : [process.cwd()],
    fileFiltering: {
      defaultExcludes: DEFAULT_EXCLUDE_PATTERNS,
      allowedExtensions: DEFAULT_ALLOWED_EXTENSIONS,
      forceTextFiles: true
    },
    fuzzyMatching: {
      maxDistanceRatio: 0.25,
      minSimilarity: 0.7,
      caseSensitive: false,
      ignoreWhitespace: true,
      preserveLeadingWhitespace: 'auto'
    },
    logging: {
      level: 'info',
      performance: false
    },
    concurrency: {
      maxConcurrentEdits: 10,
      maxGlobalConcurrentEdits: 20
    },
    limits: {
      maxReadBytes: 5 * 1024 * 1024,
      maxWriteBytes: 5 * 1024 * 1024
    }
  });
}
