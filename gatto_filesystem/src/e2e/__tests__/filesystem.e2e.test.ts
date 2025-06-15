import path from 'path';
import fs from 'fs';
import type { ChildProcess } from 'child_process';

// Używamy require zamiast import dla SDK, ponieważ TypeScript nie może znaleźć deklaracji typów
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { Client } = require('@modelcontextprotocol/sdk/client');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio');

// Zwiększamy timeout, aby dać czas na kompilację i start serwera
jest.setTimeout(30000);

describe('MCP Filesystem – E2E (Stdio)', () => {
  let transport: any; // Używamy any, ponieważ dokładna struktura StdioClientTransport jest nieznana
  let client: typeof Client;

  beforeAll(async () => {
    const projectRoot = process.cwd();
    const serverPath = path.join(projectRoot, 'src/server/index.ts');
    const tsxBin = path.join(projectRoot, 'node_modules/.bin/tsx');
    
    console.log('Project root:', projectRoot);
    console.log('Server path:', serverPath);
    console.log('TSX binary path:', tsxBin);
    console.log('TSX exists:', fs.existsSync(tsxBin));

    // Konfigurujemy nasłuchiwanie stderr bezpośrednio w opcjach transportu
    const stderrChunks: Buffer[] = [];
    
    transport = new StdioClientTransport({
      command: tsxBin,
      args: [serverPath],
      stderr: 'pipe',
      onStderr: (chunk: Buffer) => {
        stderrChunks.push(chunk);
        console.error(`[SERVER STDERR]: ${chunk.toString()}`);
      }
    });
    
    console.log('Transport created successfully');

    client = new Client({ name: 'e2e-test-client', version: '0.0.0' });
    await client.connect(transport);
  });

  afterAll(async () => {
    try {
      if (client && client.isConnected) {
        await client.close();
      }
    } catch {/* ignore */ }
    if (transport?.close) {
      await transport.close();
    }
  });

  it('initializes and lists available tools', async () => {
    console.log('Client connected successfully. Setting up timeout protection...');
    
    // Implementacja race z timeoutem dla bezpieczeństwa
    async function withTimeout<T>(promise: Promise<T>, timeoutMs: number, name: string): Promise<T> {
      let timeoutId: NodeJS.Timeout;
      const timeoutPromise = new Promise<never>((_, reject) => {
        timeoutId = setTimeout(() => {
          reject(new Error(`Operation ${name} timed out after ${timeoutMs}ms`));
        }, timeoutMs);
      });
      
      try {
        const result = await Promise.race([promise, timeoutPromise]);
        clearTimeout(timeoutId!);
        return result as T;
      } catch (error) {
        clearTimeout(timeoutId!);
        console.error(`Error in ${name}:`, error);
        throw error;
      }
    }
    
    try {
      // Sprawdź czy klient jest poprawnie zainicjalizowany
      console.log('Checking client state before request:', { 
        isConnected: client.isConnected,
        transport: transport ? 'Transport exists' : 'Transport is undefined'
      });
      
      console.log('Sending tools/list request with 5s timeout...');
      const listToolsResp = await withTimeout(
        client.request('tools/list', {}),
        5000,
        'tools/list request'
      );
      
      console.log('Tools response:', JSON.stringify(listToolsResp, null, 2));

      // Poprawiona asercja sprawdzająca pole 'tools' w odpowiedzi
      const tools = (listToolsResp as any)?.tools;
      console.log('Tools found:', tools ? tools.length : 'undefined');
      expect(tools).toBeDefined();
      expect(tools).toEqual(expect.arrayContaining([
        expect.objectContaining({ name: 'read_file' }),
        expect.objectContaining({ name: 'write_file' }),
        expect.objectContaining({ name: 'list_directory' }),
      ]));
    } catch (error) {
      console.error('Error calling tools/list:', error);
      throw error;
    } finally {
      console.log('Test completed (either success or failure)');
    }
  });
});
