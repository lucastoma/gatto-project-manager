const { spawn } = require('child_process');
const path = require('path');

// Start the server
const server = spawn('node', [path.join(__dirname, 'dist/server/index.js')], {
  stdio: ['pipe', 'pipe', 'pipe']
});

// Handle server output
server.stdout.on('data', (data) => {
  console.log(`Server: ${data}`);
});

server.stderr.on('data', (data) => {
  console.error(`Server Error: ${data}`);
});

// Send test requests after server starts
setTimeout(() => {
  testFileFiltering();
}, 2000);

async function testFileFiltering() {
  try {
    // Test reading allowed OpenCL file
    await sendRequest('read_file', {
      path: path.join(__dirname, 'test.cl'),
      encoding: 'auto'
    });
    
    // Test reading excluded file (in dist directory)
    try {
      await sendRequest('read_file', {
        path: path.join(__dirname, 'dist/server/index.js'),
        encoding: 'auto'
      });
    } catch (error) {
      console.log('Correctly blocked excluded file:', error);
    }
    
    // Test reading disallowed extension when forceTextFiles is true
    try {
      await sendRequest('read_file', {
        path: path.join(__dirname, 'test.exe'),
        encoding: 'auto'
      });
    } catch (error) {
      console.log('Correctly blocked disallowed extension:', error);
    }
    
  } catch (error) {
    console.error('Test failed:', error);
  } finally {
    server.kill();
  }
}

function sendRequest(tool, args) {
  return new Promise((resolve, reject) => {
    const requestId = Date.now();
    const request = JSON.stringify({
      jsonrpc: '2.0',
      method: 'call_tool',
      params: {
        name: tool,
        args
      },
      id: requestId
    }) + '\n';
    
    server.stdin.write(request);
    
    const listener = (data) => {
      const response = data.toString().trim();
      try {
        const json = JSON.parse(response);
        if (json.id === requestId) {
          server.stdout.off('data', listener);
          if (json.error) {
            reject(json.error);
          } else {
            resolve(json.result);
          }
        }
      } catch (e) {
        server.stdout.off('data', listener);
        reject(e);
      }
    };
    
    server.stdout.on('data', listener);
  });
}
