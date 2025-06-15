/// <reference types="jest" />
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { applyFileEdits, FuzzyMatchConfig } from '../fuzzyEdit';
import * as binaryDetect from '../../utils/binaryDetect';
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

    it('successfully edits a text file containing NUL bytes (POTENTIAL_TEXT_WITH_CAVEATS)', async () => {
        const originalContent = 'Line one\nText with a NUL\x00byte here\nLine three';
        const filePath = await createTempFile(originalContent);

        const edits = [{ oldText: 'NUL\x00byte here', newText: 'null character was here', forcePartialMatch: false }];
        const fuzzyConfig: FuzzyMatchConfig = {
            maxDistanceRatio: 0.25,
            minSimilarity: 0.7,
            caseSensitive: false,
            ignoreWhitespace: false, // Important for NUL byte matching
            preserveLeadingWhitespace: 'auto',
            debug: false
        };

        const warnSpy = jest.spyOn(noopLogger, 'warn');

        const res = await applyFileEdits(filePath, edits, fuzzyConfig, noopLogger, testConfig);

        expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining(`Attempting to edit file which may be binary or contains NUL bytes: ${filePath}`));
        expect(res.modifiedContent).toBe('Line one\nText with a null character was here\nLine three');
        
        warnSpy.mockRestore();
    });

    it('rejects editing a file classified as CONFIRMED_BINARY', async () => {
        const filePath = await createTempFile('fake binary content');
        // Mock classifyFileType to return CONFIRMED_BINARY for this specific file path or any
        const classifySpy = jest.spyOn(binaryDetect, 'classifyFileType').mockReturnValue(binaryDetect.FileType.CONFIRMED_BINARY);

        const edits = [{ oldText: 'fake', newText: 'hacked', forcePartialMatch: false }];
        const fuzzyConfig: FuzzyMatchConfig = {
            maxDistanceRatio: 0.25, minSimilarity: 0.7, caseSensitive: false, ignoreWhitespace: true, preserveLeadingWhitespace: 'auto', debug: false
        } as FuzzyMatchConfig;

        try {
            await applyFileEdits(filePath, edits, fuzzyConfig, noopLogger, testConfig);
            throw new Error('applyFileEdits should have thrown for a binary file');
        } catch (e) {
            const error = e as StructuredError;
            expect(error.code).toBe('BINARY_FILE_ERROR');
            expect(error.message).toContain('Cannot edit confirmed binary files');
        }
        classifySpy.mockRestore();
    });

    it('rejects editing a file exceeding maxReadBytes limit', async () => {
        const smallLimitConfig = {
            ...testConfig,
            limits: { ...testConfig.limits, maxReadBytes: 10 }
        } as Config;
        const originalContent = 'This content is definitely longer than ten bytes.';
        const filePath = await createTempFile(originalContent);

        const edits = [{ oldText: 'content', newText: 'stuff', forcePartialMatch: false }];
        const fuzzyConfig: FuzzyMatchConfig = {
            maxDistanceRatio: 0.25, minSimilarity: 0.7, caseSensitive: false, ignoreWhitespace: true, preserveLeadingWhitespace: 'auto', debug: false
        } as FuzzyMatchConfig;

        try {
            await applyFileEdits(filePath, edits, fuzzyConfig, noopLogger, smallLimitConfig);
            throw new Error('applyFileEdits should have thrown for exceeding maxReadBytes');
        } catch (e) {
            const error = e as StructuredError;
            expect(error.code).toBe('FILE_TOO_LARGE');
            expect(error.message).toContain('File exceeds maximum readable size');
        }
    });
});
