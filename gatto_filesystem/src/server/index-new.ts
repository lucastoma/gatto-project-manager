// Kompletny przepis serwera MCP - sprawdzamy czy plik nie jest zepsuty
// Usuń wszystko i wstaw prawidłowy kod

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { z } from 'zod';

console.error('🚀 MCP Server - checking imports...');