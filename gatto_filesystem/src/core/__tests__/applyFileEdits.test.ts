/// <reference types="jest" />
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { applyFileEdits, FuzzyMatchConfig } from '../fuzzyEdit';
import { StructuredError } from '../../types/errors';
import type { Config } from '../../server/config';

// Minimal stub config for tests
const testConfig = {
    allowedDirectories: [],
    fileFiltering: {
        defaultExcludes: [],
        allowedExtensions: ['*.txt'],
        forceTextFiles: true
    },
    fuzzyMatching: {
        maxDistanceRatio: 0.25,
        minSimilarity: 0.7,
        caseSensitive: false,
        ignoreWhitespace: true,
        preserveLeadingWhitespace: 'auto'
    },
    logging: { level: 'info', performance: false },
    concurrency: { maxConcurrentEdits: 1, maxGlobalConcurrentEdits: 1 },
    limits: { maxReadBytes: 1024 * 1024, maxWriteBytes: 1024 * 1024 }
} as unknown as Config;

// Simple no-op logger stub
const noopLogger = {
    info: () => { },
    warn: () => { },
    error: () => { },
    debug: () => { },
    trace: () => { }
} as any;

async function createTempFile(content: string): Promise<string> {
    const tmpDir = os.tmpdir();
    const filePath = path.join(tmpDir, `applyFileEdits_${Date.now()}_${Math.random().toString(36).slice(2)}.txt`);
    await fs.writeFile(filePath, content, 'utf-8');
    return filePath;
}

describe('applyFileEdits', () => {
    it('applies exact text replacement', async () => {
        const original = 'foo bar baz';
        const filePath = await createTempFile(original);

        const edits = [{ oldText: 'bar', newText: 'qux', forcePartialMatch: false }];
        const fuzzyConfig: FuzzyMatchConfig = {
            maxDistanceRatio: 0.25,
            minSimilarity: 0.7,
            caseSensitive: false,
            ignoreWhitespace: true,
            preserveLeadingWhitespace: 'auto',
            debug: false
        };

        const res = await applyFileEdits(filePath, edits, fuzzyConfig, noopLogger, testConfig);

        expect(res.modifiedContent).toBe('foo qux baz');
        expect(res.formattedDiff).toContain('-foo bar baz');
        expect(res.formattedDiff).toContain('+foo qux baz');
    });

    it('respects caseSensitive flag', async () => {
        const original = 'Hello World';
        const filePath = await createTempFile(original);

        const edits = [{ oldText: 'hello', newText: 'Hi', forcePartialMatch: false }];
        const fuzzyConfig: FuzzyMatchConfig = {
            maxDistanceRatio: 0.25,
            minSimilarity: 0.7,
            caseSensitive: true,
            ignoreWhitespace: true,
            preserveLeadingWhitespace: 'auto',
            debug: false
        };

        try {
            await applyFileEdits(filePath, edits, fuzzyConfig, noopLogger, testConfig);
            // If it reaches here, the test should fail because an error was expected
            throw new Error('applyFileEdits should have thrown an error for caseSensitive no match');
        } catch (e) {
            const error = e as StructuredError;
            expect(error.code).toBe('NO_MATCH_FOUND');
            expect(error.message).toBe('No case-sensitive match found for "hello"');
        }
    });

    it('handles ignoreWhitespace option', async () => {
        const original = 'alpha    beta';
        const filePath = await createTempFile(original);

        const edits = [{ oldText: 'alpha beta', newText: 'gamma', forcePartialMatch: false }];
        const fuzzyConfig: FuzzyMatchConfig = {
            maxDistanceRatio: 0.3,
            minSimilarity: 0.6,
            caseSensitive: false,
            ignoreWhitespace: true,
            preserveLeadingWhitespace: 'auto',
            debug: false
        };

        const res = await applyFileEdits(filePath, edits, fuzzyConfig, noopLogger, testConfig);
        expect(res.modifiedContent).toBe('gamma');
    });
});
