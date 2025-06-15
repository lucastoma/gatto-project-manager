#!/usr/bin/env node

// Napraw debug skrypt dla MCP serwera
import { spawn } from 'child_process';

const allowedDirs = ['/home/lukasz/projects/', '/home/lukasz/'];

console.log('🔧 Testowanie MCP Filesystem Server (TypeScript)...');
console.log(`📂 Allowed directories: ${allowedDirs.join(', ')}`);

const server = spawn('npx', ['tsx', 'src/server/index.ts', ...allowedDirs], {
	stdio: ['pipe', 'pipe', 'inherit']
});

// Test initialize
const initRequest = {
	jsonrpc: "2.0",
	method: "initialize",
	params: {
		protocolVersion: "2024-11-05",
		capabilities: {},
		clientInfo: { name: "test", version: "1.0.0" }
	},
	id: 1
};

console.log('📤 Wysyłam initialize request...');
server.stdin.write(JSON.stringify(initRequest) + '\n');

// Test list_directory po krótkiej przerwie
setTimeout(() => {
	const listRequest = {
		jsonrpc: "2.0",
		method: "tools/call",
		params: {
			name: "list_directory",
			arguments: {
				path: "/home/lukasz/projects/gatto-ps-ai/mcp-test"
			}
		},
		id: 2
	};

	console.log('📤 Wysyłam list_directory request...');
	server.stdin.write(JSON.stringify(listRequest) + '\n');
}, 1000);

server.stdout.on('data', (data) => {
	console.log('📥 Server response:', data.toString());
});

server.on('error', (error) => {
	console.error('💥 Server error:', error);
});

server.on('exit', (code) => {
	console.log(`🔚 Server exited with code: ${code}`);
	process.exit(code);
});

// Kill po 5 sekundach
setTimeout(() => {
	console.log('⏰ Timeout - killing server');
	server.kill();
}, 5000);