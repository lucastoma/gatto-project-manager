import path from 'node:path';
import { minimatch } from 'minimatch';
import type { Config } from '../server/config';
import { DEFAULT_EXCLUDE_PATTERNS, DEFAULT_ALLOWED_EXTENSIONS } from '../constants/extensions';

/** Zwraca true, jeśli ścieżka powinna być pominięta (wykluczona). */
export function shouldSkipPath(filePath: string, config: Config): boolean {
    const baseDir = config.allowedDirectories[0] ?? process.cwd();
    const rel = path.relative(baseDir, filePath);

    // 1) wzorce global-exclude (połącz domyślne + z configu)
    const allExcludes = [...DEFAULT_EXCLUDE_PATTERNS, ...config.fileFiltering.defaultExcludes];
    if (allExcludes.some(p => minimatch(rel, p, { dot: true, nocase: true }))) return true;

    // 2) rozszerzenia, jeśli forceTextFiles aktywne
    if (config.fileFiltering.forceTextFiles) {
        const ext = path.extname(filePath).toLowerCase();
        // Only apply extension filtering if there IS an extension.
        // Directories (ext === '') or files without extensions should not be skipped by this rule.
        if (ext !== '') {
            const allowed = [...DEFAULT_ALLOWED_EXTENSIONS, ...config.fileFiltering.allowedExtensions];
            if (!allowed.some(p => minimatch(`*${ext}`, p, { nocase: true }))) {
                return true; // Skip if extension is present but not in allowed list
            }
        }
    }
    return false;
}