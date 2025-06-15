#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { SseServerTransport } from '@modelcontextprotocol/sdk/server/sse.js'; // New import - Renamed and path changed
import express from 'express'; // New import
import * as http from 'node:http'; // New import for type
import * as pino from 'pino';
import { promises as fs } from 'node:fs';
import fsSync from 'node:fs';
import path from 'node:path';

import { getConfig } from './config.js';
import { setupToolHandlers } from '../core/toolHandlers.js';
// import * as schemas from '../core/schemas.js'; // This line seems unused, can be kept or removed
import { expandHome, normalizePath } from '../utils/pathUtils.js';

let runningServer: Server | undefined;
let httpServer: http.Server | undefined; // New global for HTTP server instance

async function shutdown(signal: NodeJS.Signals, logger: pino.Logger) {
  logger.info({ signal }, 'Received termination signal, shutting down gracefully');
  try {
    if (runningServer) {
      // await runningServer.disconnect(); // Not supported by SDK
    }
    if (httpServer) {
      await new Promise<void>((resolve, reject) => {
        httpServer!.close((err) => { // Added null assertion as httpServer will be defined if this branch is hit
          if (err) {
            logger.error({ err }, 'Error closing HTTP server');
            reject(err);
            return;
          }
          logger.info('HTTP server closed.');
          resolve();
        });
      });
    }
  } catch (err) {
    logger.error({ err }, 'Error during graceful shutdown');
  } finally {
    // Give some time for logs and server to close
    setTimeout(() => process.exit(0), 500); // Increased timeout
  }
}

async function main() {
  const config = await getConfig(process.argv.slice(2)); // Pass CLI args to getConfig

  const logsDir = path.join(process.cwd(), 'logs');
  try {
    await fs.mkdir(logsDir, { recursive: true });
  } catch (err) {
    // Log to console as logger might not be set up yet
    console.error('Could not create logs directory:', err);
  }

  const fileStream = fsSync.createWriteStream(path.join(logsDir, 'mcp-filesystem.log'), { flags: 'a' });

  const logger = pino.pino(
    {
      level: config.logging.level,
      timestamp: () => `,"timestamp":"${new Date().toISOString()}"`, // Ensure correct JSON formatting for timestamp
      base: { service: 'mcp-filesystem-server', version: '0.7.0' } // Updated version or make dynamic
    },
    pino.multistream([
      { stream: process.stdout },
      { stream: fileStream, level: 'info' } // Ensure fileStream is correctly initialized
    ])
  );

  // Logger wrappers (ensure these are robust)
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
        logger.error({ directory: dir }, `Error: ${dir} is not a directory`); // Pass object to logger
        process.exit(1);
      }
    } catch (error) {
      logger.error({ error, directory: dir }, `Error accessing directory ${dir}`);
      process.exit(1);
    }
  }));

  const server = new Server(
    { name: 'secure-filesystem-server', version: '0.7.0' }, // Ensure version is consistent or dynamic
    { capabilities: { tools: {} } }
  );

  runningServer = server;
  setupToolHandlers(server, allowedDirectories, logger, config);

  const transportType = process.env.MCP_SERVER_TRANSPORT || 'stdio'; // Default to stdio
  const serverPort = parseInt(process.env.MCP_SERVER_PORT || '3001', 10);

  if (transportType.toLowerCase() === 'http') {
    const app = express();
    app.use(express.json());

    const httpTransport = new SseServerTransport({ // Renamed
      // For E2E tests, a simple setup without session management initially.
      // Session ID generator can be undefined for stateless or simple stateful.
      // onsessioninitialized and onclose can be added if session management becomes complex.
    });

    await server.connect(httpTransport);

    app.post('/mcp', async (req, res) => {
      try {
        await httpTransport.handleRequest(req, res, req.body);
      } catch (e: any) {
        logger.error({ err: e, path: req.path, method: req.method }, "Error handling HTTP POST request");
        if (!res.headersSent) {
          res.status(500).json({ error: "Internal server error" });
        }
      }
    });

    app.get('/mcp', async (req, res) => {
      try {
        await httpTransport.handleRequest(req, res); // For SSE
      } catch (e: any) {
        logger.error({ err: e, path: req.path, method: req.method }, "Error handling HTTP GET request");
        if (!res.headersSent) {
          // SSE might have already set headers, so check
          res.status(500).json({ error: "Internal server error" });
        }
      }
    });

    app.delete('/mcp', async (req, res) => {
      try {
        await httpTransport.handleRequest(req, res); // For session termination, if supported by transport
      } catch (e: any) {
        logger.error({ err: e, path: req.path, method: req.method }, "Error handling HTTP DELETE request");
        if (!res.headersSent) {
          res.status(500).json({ error: "Internal server error" });
        }
      }
    });

    httpServer = app.listen(serverPort, () => {
      logger.info({ version: '0.7.0', transport: 'HTTP', port: serverPort, allowedDirectories, config }, 'Enhanced MCP Filesystem Server started via HTTP');
    });

    httpServer.on('error', (err) => {
      logger.error({ err }, "HTTP server error");
      process.exit(1);
    });

  } else {
    const stdioTransport = new StdioServerTransport();
    await server.connect(stdioTransport);
    logger.info({ version: '0.7.0', transport: 'Stdio', allowedDirectories, config }, 'Enhanced MCP Filesystem Server started via Stdio');
  }

  process.once('SIGINT', (sig) => shutdown(sig, logger));
  process.once('SIGTERM', (sig) => shutdown(sig, logger));
}

main().catch((error) => {
  // Use console.error as logger might not be initialized or might have failed
  console.error('Fatal error running server:', error);
  process.exit(1);
});