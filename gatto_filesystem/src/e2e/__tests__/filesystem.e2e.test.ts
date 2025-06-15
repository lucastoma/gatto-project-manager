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
        toolName: "read_file",
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

  describe('list_directory tool', () => {
    const testDirName = 'list_directory_test_root';
    const testDir = path.join(process.cwd(), testDirName); // Assuming tests run from project root

    beforeAll(async () => {
      // Create a controlled test directory structure
      if (fs.existsSync(testDir)) {
        await fs.promises.rm(testDir, { recursive: true, force: true });
      }
      await fs.promises.mkdir(testDir, { recursive: true });
      await fs.promises.writeFile(path.join(testDir, 'file1.txt'), 'content1');
      await fs.promises.writeFile(path.join(testDir, 'file2.ts'), 'content2'); // This might be filtered by default
      await fs.promises.mkdir(path.join(testDir, 'sub_dir'), { recursive: true });
      await fs.promises.writeFile(path.join(testDir, 'sub_dir', 'sub_file.txt'), 'sub_content');
      await fs.promises.mkdir(path.join(testDir, 'sub_dir', 'nested_dir'), { recursive: true });
      await fs.promises.writeFile(path.join(testDir, 'sub_dir', 'nested_dir', 'deep_file.md'), 'deep_content');
      await fs.promises.mkdir(path.join(testDir, 'empty_dir'), { recursive: true });
    });

    afterAll(async () => {
      await fs.promises.rm(testDir, { recursive: true, force: true });
    });

    it('should list directory contents non-recursively', async () => {
      const response = await runServerCommand({
        jsonrpc: '2.0',
        id: 'list-non-recursive',
        method: 'tools/call',
        params: {
          toolName: 'list_directory',
          arguments: { path: testDirName, recursive: false },
        },
      });
      expect(response.id).toBe('list-non-recursive');
      expect(response.result).toBeDefined();
      expect(response.result.result.entries).toBeDefined();
      const entries = response.result.result.entries;
      expect(entries).toEqual(expect.arrayContaining([
        expect.objectContaining({ name: 'file1.txt', type: 'file', path: 'file1.txt' }),
        // file2.ts is excluded by default config in this test setup if defaultExcludes = ['*.ts']
        // expect.objectContaining({ name: 'file2.ts', type: 'file', path: 'file2.ts' }), 
        expect.objectContaining({ name: 'sub_dir', type: 'directory', path: 'sub_dir' }),
        expect.objectContaining({ name: 'empty_dir', type: 'directory', path: 'empty_dir' }),
      ]));
      // Check that sub_file.txt is NOT present (non-recursive)
      expect(entries.find((e:any) => e.path === 'sub_dir/sub_file.txt')).toBeUndefined();
      // Verify filtered items are not present if applicable by your test config
      // For example, if *.ts is filtered by default test config:
      expect(entries.find((e:any) => e.name === 'file2.ts')).toBeUndefined(); 
    });

    it('should list directory contents recursively', async () => {
      const response = await runServerCommand({
        jsonrpc: '2.0',
        id: 'list-recursive',
        method: 'tools/call',
        params: {
          toolName: 'list_directory',
          arguments: { path: testDirName, recursive: true },
        },
      });
      expect(response.id).toBe('list-recursive');
      expect(response.result).toBeDefined();
      const entries = response.result.result.entries;
      expect(entries).toEqual(expect.arrayContaining([
        expect.objectContaining({ name: 'file1.txt', type: 'file', path: 'file1.txt' }),
        // file2.ts (still assuming filtered by default test config)
        expect.objectContaining({ name: 'sub_dir', type: 'directory', path: 'sub_dir' }),
        expect.objectContaining({ name: 'sub_file.txt', type: 'file', path: 'sub_dir/sub_file.txt' }),
        expect.objectContaining({ name: 'nested_dir', type: 'directory', path: 'sub_dir/nested_dir' }),
        expect.objectContaining({ name: 'deep_file.md', type: 'file', path: 'sub_dir/nested_dir/deep_file.md' }),
        expect.objectContaining({ name: 'empty_dir', type: 'directory', path: 'empty_dir' }),
      ]));
      // Ensure order by path
      const paths = entries.map((e:any) => e.path);
      expect(paths).toEqual([...paths].sort());
      // Verify filtered items are not present if applicable by your test config
      expect(entries.find((e:any) => e.name === 'file2.ts')).toBeUndefined(); 
    });

    it('should list empty directory recursively', async () => {
      const response = await runServerCommand({
        jsonrpc: '2.0',
        id: 'list-empty-recursive',
        method: 'tools/call',
        params: {
          toolName: 'list_directory',
          arguments: { path: `${testDirName}/empty_dir`, recursive: true },
        },
      });
      expect(response.id).toBe('list-empty-recursive');
      expect(response.result.result.entries).toEqual([]);
    });

    it('should return an error for non-existent directory when listing', async () => {
      const response = await runServerCommand({
        jsonrpc: '2.0',
        id: 'list-non-existent',
        method: 'tools/call',
        params: {
          toolName: 'list_directory',
          arguments: { path: 'non_existent_dir123', recursive: false },
        },
      });
      expect(response.id).toBe('list-non-existent');
      expect(response.error).toBeDefined();
      expect(response.error.code).toBe(-32001); // ACCESS_DENIED (Path does not exist)
      expect(response.error.message).toContain('Path does not exist or is not accessible');
    });

    // Test to ensure original functionality of checking project root (like package.json) is still possible if needed
    // This test runs on the actual project root, not the temporary testDir.
    it('can list project root and find package.json and src (non-recursive)', async () => {
      const response = await runServerCommand({
        jsonrpc: '2.0',
        id: 'list-project-root',
        method: 'tools/call',
        params: {
          toolName: 'list_directory',
          arguments: { path: '.', recursive: false }, // path: '.' refers to CWD of server, which is project root
        },
      });
      expect(response.id).toBe('list-project-root');
      expect(response.result).toBeDefined();
      const entries = response.result.result.entries;
      expect(entries.find((e:any) => e.name === 'package.json' && e.type === 'file')).toBeDefined();
      expect(entries.find((e:any) => e.name === 'src' && e.type === 'directory')).toBeDefined();
    });
  });

  it('can call server_stats tool', async () => {
    const request = {
      jsonrpc: "2.0",
      method: "tools/call",
      params: {
        toolName: "server_stats",
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
