#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');

// Test the smart_search tool
const testSmartSearch = async () => {
	console.log('Testing smart_search tool...');

	const server = spawn('node', ['dist/server/index.js'], {
		cwd: '/home/lukasz/projects/gatto-ps-ai/gatto_filesystem',
		stdio: ['pipe', 'pipe', 'pipe']
	});

	// Test smart_search request
	const request = {
		jsonrpc: '2.0',
		id: 1,
		method: 'tools/call',
		params: {
			name: 'smart_search',
			arguments: {
				query: 'basic_transfer',
				search_mode: 'hybrid',
				base_paths: ['/home/lukasz/projects/gatto-ps-ai/app/algorithms'],
				max_results: 10
			}
		}
	};

	// Send initialization first
	const initRequest = {
		jsonrpc: '2.0',
		id: 0,
		method: 'initialize',
		params: {
			protocolVersion: '2024-11-05',
			capabilities: {},
			clientInfo: { name: 'test', version: '1.0.0' }
		}
	};

	let output = '';

	server.stdout.on('data', (data) => {
		output += data.toString();
		console.log('Server output:', data.toString());
	});

	server.stderr.on('data', (data) => {
		console.error('Server error:', data.toString());
	});

	server.on('close', (code) => {
		console.log(`Server closed with code ${code}`);
		console.log('Full output:', output);
	});

	// Send initialization
	server.stdin.write(JSON.stringify(initRequest) + '\n');

	// Wait a bit then send the smart_search request
	setTimeout(() => {
		server.stdin.write(JSON.stringify(request) + '\n');

		// Close after timeout
		setTimeout(() => {
			server.kill();
		}, 3000);
	}, 1000);
};

testSmartSearch().catch(console.error);
