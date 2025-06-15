const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');

console.log('Current directory:', process.cwd());
console.log('__dirname:', __dirname);

// Look for tsx binary
const possiblePaths = [
  path.join(__dirname, 'node_modules/.bin/tsx'),
  path.join(__dirname, '../node_modules/.bin/tsx'),
  path.join(__dirname, '../../node_modules/.bin/tsx'),
  '/usr/local/bin/tsx',
  '/usr/bin/tsx'
];

console.log('Checking for tsx binary:');
for (const binPath of possiblePaths) {
  console.log(`- ${binPath}: ${fs.existsSync(binPath) ? 'EXISTS' : 'NOT FOUND'}`);
}

// Check if server built file exists
const serverPath = path.join(__dirname, 'dist/server/index.js');
console.log('Server built path:', serverPath);
console.log('Server file exists:', fs.existsSync(serverPath));

// Check SDK installation
console.log('\nChecking SDK installation:');
try {
  console.log(execSync('npm ls @modelcontextprotocol/sdk', { encoding: 'utf8' }));
} catch (error) {
  console.error('Error checking SDK:', error.message);
}

// Try running the test with more verbose output
console.log('\nTrying to run test with more debug info:');
try {
  execSync('jest src/e2e/__tests__/filesystem.e2e.test.ts --no-color --verbose', { 
    stdio: 'inherit',
    env: { ...process.env, DEBUG: '*' }
  });
} catch (error) {
  console.log('Test failed as expected, but we got debug output');
}
