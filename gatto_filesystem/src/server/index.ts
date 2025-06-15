#!/usr/bin/env node

// ZMIANA: Używamy require dla modułów SDK, aby zapewnić spójne i niezawodne ładowanie.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { Server } = require('@modelcontextprotocol/sdk/server');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio');

// Pozostałe importy (lokalne i z innych paczek) pozostają jako 'import'.
import * as pino from 'pino';
import { promises as fs } from 'node:fs';
import fsSync from 'node:fs';
import path from 'node:path';

import { getConfig } from './config';
import { setupToolHandlers } from '../core/toolHandlers';
import { expandHome, normalizePath } from '../utils/pathUtils';

async function main() {
  const config = await getConfig(process.argv.slice(2));

  const logsDir = path.join(process.cwd(), 'logs');
  try {
    await fs.mkdir(logsDir, { recursive: true });
  } catch (err) {
    console.error('Could not create logs directory:', err);
  }

  const fileStream = fsSync.createWriteStream(path.join(logsDir, 'mcp-filesystem.log'), { flags: 'a' });

  const logger = pino.pino(
    {
      level: config.logging.level,
      timestamp: () => `,"timestamp":"${new Date().toISOString()}"`,
      base: { service: 'mcp-filesystem-server', version: '0.7.0' }
    },
    pino.multistream([
      { stream: process.stderr },
      { stream: fileStream, level: 'info' }
    ])
  );

  const logWithPaths = (logFn: pino.LogFn) => (objOrMsg: any, ...args: any[]) => {
    let logObject: any = {};
    if (typeof objOrMsg === 'string') {
      logObject.msg = objOrMsg;
    } else {
      logObject = { ...objOrMsg };
    }
    if (config.logging.level === 'debug') {
      if (logObject.path && typeof logObject.path === 'string') {
        logObject.absolutePath = normalizePath(path.resolve(logObject.path));
      }
      if (logObject.directory && typeof logObject.directory === 'string') {
        logObject.absoluteDirectory = normalizePath(path.resolve(logObject.directory));
      }
    }
    return logFn(logObject, ...args);
  };
  logger.info = logWithPaths(logger.info.bind(logger));
  logger.debug = logWithPaths(logger.debug.bind(logger));
  logger.error = logWithPaths(logger.error.bind(logger));
  logger.warn = logWithPaths(logger.warn.bind(logger));

  const allowedDirectories = config.allowedDirectories.map((dir: string) => normalizePath(path.resolve(expandHome(dir))));

  await Promise.all(allowedDirectories.map(async (dir: string) => {
    try {
      const stats = await fs.stat(dir);
      if (!stats.isDirectory()) {
        logger.error({ directory: dir }, `Error: ${dir} is not a directory`);
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

  const transport = new StdioServerTransport();
  await server.connect(transport);
  logger.info({ version: '0.7.0', transport: 'Stdio', allowedDirectories, config }, 'Enhanced MCP Filesystem Server started via Stdio');

  logger.info('Server is running. Press Ctrl+C to exit.');

  process.once('SIGINT', () => process.exit(0));
  process.once('SIGTERM', () => process.exit(0));
}

main().catch((error) => {
  console.error('Fatal error running server:', error);
  process.exit(1);
});