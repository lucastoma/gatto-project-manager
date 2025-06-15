// JavaScript test script for MCP filesystem server tools

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

// Configuration
const SERVER_PATH = 'D:/projects/gatto-ps-ai-link1/gatto_filesystem_v2/dist/server/index.js';
const TEST_DIR = 'D:/projects/gatto-ps-ai-link1/mcp-test';

// Helper function to run a test
async function runTest(method, params) {
  return new Promise((resolve, reject) => {
    console.log(`\n===== Testing: ${method} =====`);
    
    const request = JSON.stringify({
      jsonrpc: '2.0',
      id: Math.floor(Math.random() * 10000),
      method,
      params
    });
    
    console.log('Request:', request);
    
    const contentLength = Buffer.byteLength(request, 'utf8');
    const header = `Content-Length: ${contentLength}\r\n\r\n`;
    const message = header + request;
    
    const node = spawn('node', [SERVER_PATH, TEST_DIR]);
    let output = '';
    
    node.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    node.stderr.on('data', (data) => {
      console.error(`Error: ${data}`);
    });
    
    node.on('close', (code) => {
      console.log('Response:');
      console.log(output);
      console.log('===========================');
      
      if (code === 0) {
        resolve(output);
      } else {
        reject(new Error(`Process exited with code ${code}`));
      }
    });
    
    node.stdin.write(message);
    node.stdin.end();
  });
}

// Run all tests
async function runAllTests() {
  try {
    console.log('\n*** STARTING MCP SERVER TESTS ***\n');
    
    // 1. Test handshake
    await runTest('handshake', {});
    
    // 2. Test list_tools
    await runTest('list_tools', {});
    
    // 3. Test read_file
    await runTest('read_file', {
      path: 'test-file.txt'
    });
    
    // 4. Test read_multiple_files
    await runTest('read_multiple_files', {
      paths: ['test-file.txt', 'test-script.js', 'subdir/nested-file.md']
    });
    
    // 5. Test write_file
    await runTest('write_file', {
      path: 'write-test.txt',
      content: 'This is a file created by the write_file tool test.'
    });
    
    // 6. Test edit_file
    await runTest('edit_file', {
      path: 'test-file.txt',
      edits: [
        {
          oldText: 'This is a test file for MCP testing.',
          newText: 'This is a modified test file for MCP testing.',
          allowMultiple: false
        }
      ],
      dryRun: true
    });
    
    // 7. Test create_directory
    await runTest('create_directory', {
      path: 'new-test-dir'
    });
    
    // 8. Test list_directory
    await runTest('list_directory', {
      path: '.'
    });
    
    // 9. Test directory_tree
    await runTest('directory_tree', {
      path: '.'
    });
    
    // 10. Test search_files
    await runTest('search_files', {
      path: '.',
      pattern: '*.txt'
    });
    
    // 11. Test get_file_info
    await runTest('get_file_info', {
      path: 'test-file.txt'
    });
    
    // 12. Test list_allowed_directories
    await runTest('list_allowed_directories', {});
    
    console.log('\nAll tests completed!');
  } catch (error) {
    console.error('Test suite failed:', error);
  }
}

// Run the tests
runAllTests();