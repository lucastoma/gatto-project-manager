import path from 'path';
import fs from 'fs';
import type { ChildProcess } from 'child_process';

// Używamy require, ponieważ jest to sprawdzona metoda ładowania tego SDK w projekcie
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { Client } = require('@modelcontextprotocol/sdk/client');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio');

jest.setTimeout(30000);

describe('MCP Filesystem – E2E (Stdio)', () => {
  let transport: any; // typ 'any' jest tu akceptowalny dla uproszczenia
  let client: typeof Client;

  beforeAll(async () => {
    const projectRoot = process.cwd();
    const serverPath = path.join(projectRoot, 'src/server/index.ts');
    const tsxBin = path.join(projectRoot, 'node_modules/.bin/tsx');

    console.log('Project root:', projectRoot);
    console.log('Server path:', serverPath);
    console.log('TSX binary path:', tsxBin);
    console.log('TSX exists:', fs.existsSync(tsxBin));

    transport = new StdioClientTransport({
      command: tsxBin,
      args: [serverPath],
      stderr: 'pipe',
    });

    if ((transport as any).proc?.stderr) {
      (transport as any).proc.stderr.on('data', (data: Buffer) => {
        console.error(`[SERVER STDERR]: ${data.toString()}`);
      });
    }

    client = new Client({ name: 'e2e-test-client', version: '0.0.0' });
    await client.connect(transport);
  });

  afterAll(async () => {
    try {
      if (client?.isConnected) {
        await client.close();
      }
    } catch {/* ignore */ }
    await transport?.close();
  });

  it('initializes and lists available tools', async () => {
    console.log('Client connected. Sending tools/list request...');

    try {
      const listToolsResp = await client.request('tools/list', {});
      console.log('Tools response:', JSON.stringify(listToolsResp, null, 2));

      const tools = (listToolsResp as any)?.tools;
      expect(tools).toBeDefined();
      expect(tools).toEqual(expect.arrayContaining([
        expect.objectContaining({ name: 'read_file' }),
        expect.objectContaining({ name: 'write_file' }),
        expect.objectContaining({ name: 'list_directory' }),
      ]));
    } catch (error) {
      console.error('Error calling tools/list:', error);
      throw error;
    }
  });
});