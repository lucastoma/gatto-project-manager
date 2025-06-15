const { spawn } = require('child_process');

async function testMCP() {
	const server = spawn('node', ['/home/lukasz/projects/gatto-ps-ai/gatto_filesystem/dist/server/index.js', '/home/lukasz/projects/gatto-ps-ai']);

	let output = '';
	let ready = false;

	server.stderr.on('data', (data) => {
		const text = data.toString();
		console.log('Server:', text);
		if (text.includes('✅ MCP Filesystem Server connected and ready!')) {
			ready = true;
			runTest();
		}
	});

	server.stdout.on('data', (data) => {
		output += data.toString();
		console.log('Response:', data.toString());
	});

	function runTest() {
		if (!ready) return;

		console.log('\n=== Testing EXACT MATCH ===');
		const testData = {
			method: "tools/call",
			params: {
				name: "edit_file",
				arguments: {
					path: "/home/lukasz/projects/gatto-ps-ai/app/algorithms/algorithm_05_lab_transfer/processor.py",
					edits: [{
						oldText: "from app.core.development_logger import get_logger",
						newText: "from app.core.development_logger import get_logger  # Test exact match"
					}],
					dryRun: true
				}
			}
		};

		server.stdin.write(JSON.stringify(testData) + '\n');

		setTimeout(() => {
			console.log('\n=== Testing HIGH SIMILARITY ===');
			const testData2 = {
				method: "tools/call",
				params: {
					name: "edit_file",
					arguments: {
						path: "/home/lukasz/projects/gatto-ps-ai/app/algorithms/algorithm_05_lab_transfer/processor.py",
						edits: [{
							oldText: "from app.core.performance_profiler import get_profler",  // typo: profler vs profiler
							newText: "from app.core.performance_profiler import get_profiler  # High similarity test"
						}],
						dryRun: true
					}
				}
			};
			server.stdin.write(JSON.stringify(testData2) + '\n');

			setTimeout(() => {
				server.kill();
			}, 2000);
		}, 2000);
	}

	server.on('close', (code) => {
		console.log(`\nServer exited with code ${code}`);
	});
}

testMCP();
