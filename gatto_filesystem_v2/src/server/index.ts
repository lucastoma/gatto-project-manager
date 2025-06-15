#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import * as pino from 'pino';
import fs from 'node:fs/promises';
import path from 'node:path';

import { loadConfig } from './config.js';
import { setupToolHandlers } from '../core/toolHandlers.js';
import * as schemas from '../core/schemas.js';
import { expandHome, normalizePath } from '../utils/pathUtils.js';

async function main() {
  const config = await loadConfig();

  const logger = pino.pino({
    level: config.logging.level,
    formatters: { level: (label: string) => ({ level: label }) },
    timestamp: () => `,"timestamp":"${new Date().toISOString()}"`,
    base: { service: 'mcp-filesystem-server', version: '0.7.0' }
  });

  const allowedDirectories = config.allowedDirectories.map(dir => normalizePath(path.resolve(expandHome(dir))));

  await Promise.all(allowedDirectories.map(async (dir) => {
    try {
      const stats = await fs.stat(dir);
      if (!stats.isDirectory()) {
        logger.error(`Error: ${dir} is not a directory`);
        process.exit(1);
      }
    } catch (error) {
      logger.error({ error, directory: dir }, `Error accessing directory ${dir}`);
      process.exit(1);
    }
  }));

  const server = new Server(
    { name: 'secure-filesystem-server', version: '0.7.0' },
    { capabilities: { tools: {} } }
  );

  setupToolHandlers(server, allowedDirectories, logger, config);
// list_tools jest już zarejestrowane w toolHandlers.ts, więc usuwamy duplikat

  const transport = new StdioServerTransport();
  await server.connect(transport);

  logger.info({ version: '0.7.0', allowedDirectories, config }, 'Enhanced MCP Filesystem Server started');
}

main().catch((error) => {
  console.error('Fatal error running server:', error);
  process.exit(1);
});
