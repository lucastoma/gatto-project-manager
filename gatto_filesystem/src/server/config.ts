import { z } from "zod";
import fs from "node:fs/promises";
import path from "node:path";

export const ConfigSchema = z.object({
  allowedDirectories: z.array(z.string()),
  fileFiltering: z.object({
    defaultExcludes: z.array(z.string()).default([
      '**/build/**',
      '**/dist/**',
      '**/node_modules/**',
      '**/.git/**',
      '**/*.jpg', '**/*.png', '**/*.gif', '**/*.pdf',
      '**/*.zip', '**/*.tar', '**/*.gz'
    ]),
    allowedExtensions: z.array(z.string()).default([
      '*.txt', '*.js', '*.jsx', '*.ts', '*.tsx', '*.json', '*.yaml', '*.yml',
      '*.html', '*.htm', '*.css', '*.scss', '*.sass', '*.less', '*.py', '*.java', '*.go',
      '*.rs', '*.rb', '*.php', '*.sh', '*.bash', '*.zsh', '*.md', '*.markdown', '*.xml',
      '*.svg', '*.csv', '*.toml', '*.ini', '*.cfg', '*.conf', '*.env', '*.ejs', '*.pug',
      '*.vue', '*.svelte', '*.graphql', '*.gql', '*.proto', '*.kt', '*.kts', '*.swift',
      '*.m', '*.h', '*.c', '*.cpp', '*.hpp', '*.cs', '*.fs', '*.fsx', '*.clj', '*.cljs',
      '*.cljc', '*.edn', '*.ex', '*.exs', '*.erl', '*.hrl', '*.lua', '*.sql', '*.pl',
      '*.pm', '*.r', '*.jl', '*.dart', '*.groovy', '*.gradle', '*.nim', '*.zig', '*.v',
      '*.vh', '*.vhd', '*.cl', '*.tex', '*.sty', '*.cls', '*.rst', '*.adoc', '*.asciidoc'
    ]),
    forceTextFiles: z.boolean().default(true)
  }).default({}),
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
            defaultExcludes: [
                '**/build/**',
                '**/dist/**',
                '**/node_modules/**',
                '**/.git/**',
                '**/*.jpg', '**/*.png', '**/*.gif', '**/*.pdf',
                '**/*.zip', '**/*.tar', '**/*.gz'
            ],
            allowedExtensions: [
                '*.txt', '*.js', '*.jsx', '*.ts', '*.tsx', '*.json', '*.yaml', '*.yml',
                '*.html', '*.htm', '*.css', '*.scss', '*.sass', '*.less', '*.py', '*.java', '*.go',
                '*.rs', '*.rb', '*.php', '*.sh', '*.bash', '*.zsh', '*.md', '*.markdown', '*.xml',
                '*.svg', '*.csv', '*.toml', '*.ini', '*.cfg', '*.conf', '*.env', '*.ejs', '*.pug',
                '*.vue', '*.svelte', '*.graphql', '*.gql', '*.proto', '*.kt', '*.kts', '*.swift',
                '*.m', '*.h', '*.c', '*.cpp', '*.hpp', '*.cs', '*.fs', '*.fsx', '*.clj', '*.cljs',
                '*.cljc', '*.edn', '*.ex', '*.exs', '*.erl', '*.hrl', '*.lua', '*.sql', '*.pl',
                '*.pm', '*.r', '*.jl', '*.dart', '*.groovy', '*.gradle', '*.nim', '*.zig', '*.v',
                '*.vh', '*.vhd', '*.cl', '*.tex', '*.sty', '*.cls', '*.rst', '*.adoc', '*.asciidoc'
            ],
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
            maxConcurrentEdits: 10
        }
    });
}
