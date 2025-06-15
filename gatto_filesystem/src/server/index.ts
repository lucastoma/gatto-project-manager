#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import * as pino from 'pino';
import { promises as fs } from 'node:fs';
import fsSync from 'node:fs';
import path from 'node:path';

import { getConfig } from './config.js';
import { setupToolHandlers } from '../core/toolHandlers.js';
import * as schemas from '../core/schemas.js';
import { expandHome, normalizePath } from '../utils/pathUtils.js';

let runningServer: Server | undefined;

async function shutdown(signal: NodeJS.Signals, logger: pino.Logger) {
  logger.info({ signal }, 'Received termination signal, shutting down gracefully');
  try {
    if (runningServer) {
      // await runningServer.disconnect(); // Not supported by SDK, just exit
    }
  } catch (err) {
    logger.error({ err }, 'Error during graceful shutdown');
  } finally {
    // Give some time for logs to flush
    setTimeout(() => process.exit(0), 100);
  }
}

async function main() {
  // TODO: parse process.argv or pass args if needed
  const config = await getConfig([]);

  // Create logs directory if it doesn't exist
  const logsDir = path.join(process.cwd(), 'logs');
  try {
    await fs.mkdir(logsDir, { recursive: true });
  } catch (err) {
    console.error('Could not create logs directory:', err);
  }

  // Create file stream for logging
  const fileStream = fsSync.createWriteStream(path.join(logsDir, 'mcp-filesystem.log'), { flags: 'a' });

  const logger = pino.pino(
    {
      level: config.logging.level,
      timestamp: () => `,"timestamp":"${new Date().toISOString()}"`,
      base: { service: 'mcp-filesystem-server', version: '0.7.0' }
    },
    pino.multistream([
      { stream: process.stdout },
      { stream: fileStream, level: 'info' }
    ])
  );

  // Add path information for debug logs
  const logWithPaths = (logFn: Function) => (obj: any) => {
    if (config.logging.level === 'debug') {
      if (obj.path && typeof obj.path === 'string') {
        obj.absolutePath = normalizePath(path.resolve(obj.path));
      }
      if (obj.directory && typeof obj.directory === 'string') {
        obj.absoluteDirectory = normalizePath(path.resolve(obj.directory));
      }
    }
    return logFn(obj);
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

  runningServer = server;

  setupToolHandlers(server, allowedDirectories, logger, config);
  // list_tools jest już zarejestrowane w toolHandlers.ts, więc usuwamy duplikat

  const transport = new StdioServerTransport();
  await server.connect(transport);

  logger.info({ version: '0.7.0', allowedDirectories, config }, 'Enhanced MCP Filesystem Server started');

  // Setup signal handlers after server is running
  process.once('SIGINT', (sig) => shutdown(sig, logger));
  process.once('SIGTERM', (sig) => shutdown(sig, logger));
}

main().catch((error) => {
  console.error('Fatal error running server:', error);
  process.exit(1);
});
