const fs = require('fs');
const path = require('path');
const { promisify } = require('util');

const readdir = promisify(fs.readdir);
const stat = promisify(fs.stat);
const readFile = promisify(fs.readFile);
const writeFile = promisify(fs.writeFile);

// RegExp dla importów z rozszerzeniem .js
const jsExtensionRegex = /from\s+['"]([^'"@][^'"]*?)\.js['"]/g;
const typeJsExtensionRegex = /from\s+['"]type:([^'"@][^'"]*?)\.js['"]/g;
const sdkJsExtensionRegex = /from\s+['"](@modelcontextprotocol\/sdk\/.+?)\.js['"]/g;

async function walkDir(dir) {
    const files = await readdir(dir);
    const results = [];

    for (const file of files) {
        const filePath = path.join(dir, file);
        const fileStat = await stat(filePath);

        if (fileStat.isDirectory() && !file.includes('node_modules') && !file.includes('dist')) {
            results.push(...await walkDir(filePath));
        } else if (fileStat.isFile() && file.endsWith('.ts')) {
            results.push(filePath);
        }
    }

    return results;
}

async function fixImports(filePath) {
    console.log(`Sprawdzanie ${filePath}...`);
    const content = await readFile(filePath, 'utf-8');

    // Usunięcie rozszerzeń .js z importów
    let newContent = content
        .replace(jsExtensionRegex, "from '$1'")
        .replace(typeJsExtensionRegex, "from 'type:$1'")
        .replace(sdkJsExtensionRegex, "from '$1'");

    // Zapisu pliku tylko jeśli jest zmieniony
    if (content !== newContent) {
        await writeFile(filePath, newContent, 'utf-8');
        console.log(`✅ Naprawiono importy w ${filePath}`);
        return true;
    }
    
    return false;
}

async function main() {
    try {
        const rootDir = path.join(__dirname, 'src');
        const files = await walkDir(rootDir);
        
        console.log(`Znaleziono ${files.length} plików TypeScript do sprawdzenia`);
        
        let modifiedCount = 0;
        for (const file of files) {
            const modified = await fixImports(file);
            if (modified) modifiedCount++;
        }
        
        console.log(`\nZakończono! Zmodyfikowano ${modifiedCount} plików`);
    } catch (error) {
        console.error('Wystąpił błąd:', error);
    }
}

main();
