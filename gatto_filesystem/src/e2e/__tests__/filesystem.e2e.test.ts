import path from 'path';
import fs from 'fs';
import { spawn, ChildProcess } from 'child_process';

jest.setTimeout(30000);

describe('MCP Filesystem – E2E (Stdio)', () => {

  async function runServerCommand(request: any): Promise<any> {
    const projectRoot = process.cwd();
    const serverPath = path.join(projectRoot, 'src/server/index.ts');
    const tsxBin = path.join(projectRoot, 'node_modules/.bin/tsx');

    return new Promise((resolve, reject) => {
      console.log(`[TEST] Sending request: ${JSON.stringify(request)}`);

      const process = spawn(tsxBin, [serverPath], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let stdout = '';
      let stderr = '';
      let foundResponse = false;
      let responseData: any = null;
      let timeoutId: NodeJS.Timeout;

      process.stdout.on('data', (data) => {
        const chunk = data.toString();
        stdout += chunk;

        // Look for JSON response lines
        const lines = chunk.split('\n');
        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('{') && trimmed.includes('"jsonrpc"') && trimmed.includes(`"id":${request.id}`)) {
            try {
              const response = JSON.parse(trimmed);
              if (!foundResponse) {
                foundResponse = true;
                responseData = response;
                clearTimeout(timeoutId);
                console.log(`[TEST] Received response: ${JSON.stringify(response)}`);
                if (!process.killed) {
                  process.kill();
                }
              }
            } catch (e) {
              console.log(`[TEST] Failed to parse JSON: ${trimmed}`);
            }
          }
        }
      });

      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      process.on('close', (code) => {
        clearTimeout(timeoutId);
        if (foundResponse && responseData) {
          resolve(responseData);
        } else {
          console.error('[TEST] Server stdout:', stdout);
          console.error('[TEST] Server stderr:', stderr);
          reject(new Error(`Process exited with code ${code}, no JSON response found. Stdout: ${stdout}, Stderr: ${stderr}`));
        }
      });

      process.on('error', (err) => {
        clearTimeout(timeoutId);
        console.error('[TEST] Process error:', err);
        reject(err);
      });

      // Send the request
      const requestStr = JSON.stringify(request) + '\n';
      process.stdin.write(requestStr);
      process.stdin.end();

      // Timeout
      timeoutId = setTimeout(() => {
        if (!foundResponse) {
          if (!process.killed) {
            process.kill();
          }
          // The 'close' event will handle rejection
        }
      }, 15000);
    });
  }

  it('responds to initialize request', async () => {
    const request = {
      jsonrpc: "2.0",
      method: "initialize",
      params: {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: { name: "test", version: "1.0.0" }
      },
      id: 1
    };

    const response = await runServerCommand(request);

    expect(response.result).toBeDefined();
    expect(response.result.serverInfo).toBeDefined();
    expect(response.result.serverInfo.name).toBe('mcp-filesystem-server');
  });

  it('responds to tools/list request', async () => {
    const request = {
      jsonrpc: "2.0",
      method: "tools/list",
      params: {},
      id: 2
    };

    const response = await runServerCommand(request);

    expect(response.result).toBeDefined();
    expect(response.result.tools).toBeDefined();
    expect(Array.isArray(response.result.tools)).toBe(true);
    expect(response.result.tools.length).toBeGreaterThan(5);

    const toolNames = response.result.tools.map((t: any) => t.name);
    expect(toolNames).toContain('read_file');
    expect(toolNames).toContain('write_file');
    expect(toolNames).toContain('list_directory');
    expect(toolNames).toContain('edit_file');
    expect(toolNames).toContain('search_files');
  });

  it('can call read_file tool', async () => {
    const request = {
      jsonrpc: "2.0",
      method: "tools/call",
      params: {
        name: "read_file",
        arguments: {
          path: "test.cl"
        }
      },
      id: 3
    };

    const response = await runServerCommand(request);

    expect(response.result).toBeDefined();
    expect(response.result.result.content).toBeDefined();
    expect(typeof response.result.result.content).toBe('string');
    expect(response.result.result.content).toContain('kernel'); // test.cl contains __kernel
  });

  it('can call list_directory tool', async () => {
    const request = {
      jsonrpc: "2.0",
      method: "tools/call",
      params: {
        name: "list_directory",
        arguments: {
          path: "."
        }
      },
      id: 4
    };

    const response = await runServerCommand(request);

    expect(response.result).toBeDefined();
    expect(response.result.result.entries).toBeDefined();
    expect(Array.isArray(response.result.result.entries)).toBe(true);

    // Should find some common files
    const entryNames = response.result.result.entries.map((e: any) => e.name);
    expect(entryNames).toContain('package.json');
    expect(entryNames).toContain('src');
  });

  it('can call server_stats tool', async () => {
    const request = {
      jsonrpc: "2.0",
      method: "tools/call",
      params: {
        name: "server_stats",
        arguments: {}
      },
      id: 5
    };

    const response = await runServerCommand(request);

    expect(response.result).toBeDefined();
    expect(response.result.result.requestCount).toBeDefined();
    expect(response.result.result.config).toBeDefined();
    expect(typeof response.result.result.requestCount).toBe('number');
  });
});
