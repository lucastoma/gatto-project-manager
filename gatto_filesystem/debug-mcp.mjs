#!/usr/bin/env node

// Test skrypt do debugowania MCP serwera
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

// Sprawdź jaki plik jest dostępny w dist
const distDir = path.join(process.cwd(), 'dist');
console.log('📁 Checking dist directory...');

try {
	const files = fs.readdirSync(distDir);
	console.log('📋 Files in dist:', files);

	// Znajdź główny plik - może być server/index.js
	let serverPath;
	if (files.includes('index.js')) {
		serverPath = path.join(distDir, 'index.js');
	} else if (files.includes('server')) {
		const serverFiles = fs.readdirSync(path.join(distDir, 'server'));
		console.log('📋 Files in dist/server:', serverFiles);
		if (serverFiles.includes('index.js')) {
			serverPath = path.join(distDir, 'server', 'index.js');
		}
	}

	if (!serverPath) {
		console.error('❌ Cannot find main server file in dist/');
		process.exit(1);
	}  // Użyj TypeScript bezpośrednio zamiast kompilowanego JS
	const serverArgs = ['tsx', 'src/server/index.ts', ...allowedDirs];

	console.log('🔧 Testowanie MCP Filesystem Server (TypeScript)...');
	console.log(`📂 Allowed directories: ${allowedDirs.join(', ')}`);

	const server = spawn('npx', serverArgs, {
		stdio: ['pipe', 'pipe', 'inherit']
	});  // Test initialize
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

	// Test list_directory
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

	// Kill after 5 seconds
	setTimeout(() => {
		console.log('⏰ Timeout - killing server');
		server.kill();
	}, 5000);

} catch (error) {
	console.error('❌ Error reading dist directory:', error);
	console.log('🔨 Próbuj uruchomić: npm run build');
	process.exit(1);
}