import fs from 'node:fs/promises';
import { createTwoFilesPatch } from 'diff';
import { classifyFileType, FileType } from '../utils/binaryDetect';
import { createError } from '../types/errors';
import { get as fastLevenshtein } from 'fast-levenshtein';
import { PerformanceTimer } from '../utils/performance';
import type { Logger } from 'pino';
import type { EditOperation } from './schemas';
import type { Config } from '../server/config';

interface AppliedEditRange {
  startLine: number;
  endLine: number;
  editIndex: number; // To identify which edit operation this range belongs to
}

function doRangesOverlap(range1: AppliedEditRange, range2: { startLine: number, endLine: number }): boolean {
  return Math.max(range1.startLine, range2.startLine) <= Math.min(range1.endLine, range2.endLine);
}

export interface FuzzyMatchConfig {
  maxDistanceRatio: number;
  minSimilarity: number;
  caseSensitive: boolean;
  ignoreWhitespace: boolean;
  preserveLeadingWhitespace: 'auto' | 'strict' | 'normalize';
  debug: boolean;
  forcePartialMatch?: boolean; // Added for forcePartialMatch option per edit
}

export interface ApplyFileEditsResult {
  modifiedContent: string;
  formattedDiff: string;
}

function normalizeLineEndings(text: string): string {
  return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
}

function createUnifiedDiff(originalContent: string, newContent: string, filepath: string = 'file'): string {
  const normalizedOriginal = normalizeLineEndings(originalContent);
  const normalizedNew = normalizeLineEndings(newContent);
  return createTwoFilesPatch(
    filepath,
    filepath,
    normalizedOriginal,
    normalizedNew,
    'original',
    'modified'
  );
}

function preprocessText(text: string, config: FuzzyMatchConfig): string {
  let processed = text;
  if (!config.caseSensitive) {
    processed = processed.toLowerCase();
  }
  if (config.ignoreWhitespace) {
    processed = processed.replace(/[ \t]+/g, ' ').replace(/\n+/g, '\n').trim();
  }
  return processed;
}

// Main edit application logic
export async function applyFileEdits(
  filePath: string,
  edits: EditOperation[],
  config: FuzzyMatchConfig,
  logger: Logger,
  globalConfig: Config
): Promise<ApplyFileEditsResult> {
  const timer = new PerformanceTimer('applyFileEdits', logger, globalConfig);
  let levenshteinIterations = 0;

  try {
    // Phase 2: File Size Limit Check (Early Exit)
    const stats = await fs.stat(filePath);
    if (stats.size > globalConfig.limits.maxReadBytes) {
      throw createError(
        'FILE_TOO_LARGE',
        `File exceeds maximum readable size of ${globalConfig.limits.maxReadBytes} bytes: ${filePath}`,
        { filePath, fileSize: stats.size, maxSize: globalConfig.limits.maxReadBytes }
      );
    }
    const buffer = await fs.readFile(filePath);

    // Phase 2: Update File Type Classification & Adjust Logic
    const fileType = classifyFileType(buffer, filePath);

    if (fileType === FileType.CONFIRMED_BINARY) {
      throw createError(
        'BINARY_FILE_ERROR',
        'Cannot edit confirmed binary files',
        { filePath, detectedAs: 'CONFIRMED_BINARY' }
      );
    } else if (fileType === FileType.POTENTIAL_TEXT_WITH_CAVEATS) {
      logger.warn(`Attempting to edit file which may be binary or contains NUL bytes: ${filePath}`);
      // Proceed with edit for POTENTIAL_TEXT_WITH_CAVEATS
    }
    // For FileType.TEXT, proceed as normal

    const originalContent = normalizeLineEndings(buffer.toString('utf-8'));
    let modifiedContent = originalContent;
    const appliedRanges: AppliedEditRange[] = [];

    validateEdits(edits, config.debug, logger);

    for (const [editIndex, edit] of edits.entries()) {
      let replaced = false;
      const normalizedOld = normalizeLineEndings(edit.oldText);
      const normalizedNew = normalizeLineEndings(edit.newText);

      // Check for exact match first as it's faster
      const exactMatchIndex = modifiedContent.indexOf(normalizedOld);

      if (exactMatchIndex !== -1) {
        // Exact match found – decide between simple in-line replacement and multi-line block replacement
        const oldLinesForIndent = normalizedOld.split('\n');
        const newLinesForIndent = normalizedNew.split('\n');

        if (oldLinesForIndent.length === 1 && newLinesForIndent.length === 1) {
          // Simple in-line (single-line) replacement. Preserve surrounding text.
          const lineNumberOfMatch = modifiedContent.slice(0, exactMatchIndex).split('\n').length - 1;

          // Overlap guard – single-line range
          const currentEditTargetRange = { startLine: lineNumberOfMatch, endLine: lineNumberOfMatch, editIndex };
          for (const appliedRange of appliedRanges) {
            if (doRangesOverlap(appliedRange, currentEditTargetRange)) {
              throw createError('OVERLAPPING_EDIT', `Edit ${editIndex + 1} (exact match) overlaps with previously applied edit ${appliedRange.editIndex + 1}.`);
            }
          }

          modifiedContent =
            modifiedContent.slice(0, exactMatchIndex) +
            normalizedNew +
            modifiedContent.slice(exactMatchIndex + normalizedOld.length);

          appliedRanges.push(currentEditTargetRange);
          replaced = true;
        } else {
          // Multi-line replacement – keep existing indentation logic
          replaced = true;
          const contentLines = modifiedContent.split('\n');
          let charCount = 0;
          let lineNumberOfMatch = 0;
          for (let i = 0; i < contentLines.length; i++) {
            if (charCount + contentLines[i].length + 1 > exactMatchIndex) {
              lineNumberOfMatch = i;
              break;
            }
            charCount += contentLines[i].length + 1;
          }
          const originalIndent = contentLines[lineNumberOfMatch].match(/^\s*/)?.[0] ?? '';
          const indentedNewLines = applyRelativeIndentation(newLinesForIndent, oldLinesForIndent, originalIndent, config.preserveLeadingWhitespace);
          const currentEditTargetRange = {
            startLine: lineNumberOfMatch,
            endLine: lineNumberOfMatch + oldLinesForIndent.length - 1,
            editIndex
          };
          for (const appliedRange of appliedRanges) {
            if (doRangesOverlap(appliedRange, currentEditTargetRange)) {
              throw createError('OVERLAPPING_EDIT', `Edit ${editIndex + 1} (exact match) overlaps with previously applied edit ${appliedRange.editIndex + 1}.`);
            }
          }
          const tempContentLines = modifiedContent.split('\n');
          tempContentLines.splice(lineNumberOfMatch, oldLinesForIndent.length, ...indentedNewLines);
          modifiedContent = tempContentLines.join('\n');
          appliedRanges.push({
            startLine: currentEditTargetRange.startLine,
            endLine: currentEditTargetRange.startLine + indentedNewLines.length - 1,
            editIndex
          });
        }

      } else if (!config.caseSensitive) {
        // Logikę FUZZY MATCHING wykonujemy TYLKO, gdy NIE jest wymagana wrażliwość na wielkość liter
        // LUB gdy wcześniejsze, prostsze metody (np. ignorowanie białych znaków) już znalazły dopasowanie.
        // Ta struktura else if jest kluczowa.

        const contentLines = modifiedContent.split('\n');
        const oldLines = normalizedOld.split('\n');
        const processedOld = preprocessText(normalizedOld, config);

        let bestMatch = { distance: Infinity, index: -1, text: '', similarity: 0, windowSize: 0 };
        for (let i = 0; i <= contentLines.length - oldLines.length; i++) {
          const windowLines = contentLines.slice(i, i + oldLines.length);
          const windowText = windowLines.join('\n');
          const processedWindowText = preprocessText(windowText, config);
          if (processedOld.length === 0 && processedWindowText.length > 0) continue;
          levenshteinIterations++;
          const distance = fastLevenshtein(processedOld, processedWindowText);
          const maxLength = Math.max(processedOld.length, processedWindowText.length);
          const similarity = maxLength > 0 ? 1 - distance / maxLength : 1;
          if (similarity > bestMatch.similarity) {
            bestMatch = { distance, index: i, text: windowText, similarity, windowSize: oldLines.length };
          }
        }

        const distanceRatio = bestMatch.text.length > 0 || normalizedOld.length > 0 ? bestMatch.distance / Math.max(bestMatch.text.length, normalizedOld.length) : 0;

        if (bestMatch.index !== -1 && bestMatch.similarity >= config.minSimilarity && distanceRatio <= config.maxDistanceRatio) {
          // Zastosuj edycję na podstawie najlepszego znalezionego dopasowania fuzzy
          const fuzzyMatchedLines = bestMatch.text.split('\n');
          const newLinesForFuzzy = normalizedNew.split('\n');
          const originalIndentFuzzy = contentLines[bestMatch.index].match(/^\s*/)?.[0] ?? '';
          const indentedNewLinesFuzzy = applyRelativeIndentation(newLinesForFuzzy, fuzzyMatchedLines, originalIndentFuzzy, config.preserveLeadingWhitespace);
          const currentFuzzyEditTargetRange = { startLine: bestMatch.index, endLine: bestMatch.index + bestMatch.windowSize - 1 };
          for (const appliedRange of appliedRanges) {
            if (doRangesOverlap(appliedRange, currentFuzzyEditTargetRange)) {
              throw createError('OVERLAPPING_EDIT', `Edit ${editIndex + 1} (fuzzy match) overlaps with previously applied edit ${appliedRange.editIndex + 1}.`);
            }
          }
          const tempContentLinesFuzzy = modifiedContent.split('\n');
          tempContentLinesFuzzy.splice(bestMatch.index, bestMatch.windowSize, ...indentedNewLinesFuzzy);
          modifiedContent = tempContentLinesFuzzy.join('\n');
          appliedRanges.push({ startLine: currentFuzzyEditTargetRange.startLine, endLine: currentFuzzyEditTargetRange.startLine + indentedNewLinesFuzzy.length - 1, editIndex });
          replaced = true;
        }
      }

      if (!replaced) {
        // No match found after all strategies – tailor error based on case sensitivity requirement
        if (config.caseSensitive) {
          throw createError('NO_MATCH_FOUND', `No case-sensitive match found for "${edit.oldText}"`);
        }
        throw createError('NO_MATCH', `No match found for edit "${edit.oldText}"`);
      }
    }

    const diff = createUnifiedDiff(originalContent, modifiedContent, filePath);
    const MAX_DIFF_LINES = 4000; // Configurable limit for diff lines
    const diffLines = diff.split('\n');
    let formattedDiff = "";
    if (diffLines.length > MAX_DIFF_LINES) {
      formattedDiff = "```diff\n" +
        diffLines.slice(0, MAX_DIFF_LINES).join('\n') +
        `\n...diff truncated (${diffLines.length - MAX_DIFF_LINES} lines omitted)\n` +
        "```\n\n";
    } else {
      formattedDiff = "```diff\n" + diff + "\n```\n\n";
    }

    timer.end({
      editsCount: edits.length,
      levenshteinIterations,
      charactersProcessed: originalContent.length
    });

    return { modifiedContent, formattedDiff };
  } catch (error) {
    timer.end({ result: 'error' });
    throw error;
  }
}

function applyRelativeIndentation(
  newLines: string[],
  oldLines: string[],
  originalIndent: string,
  preserveMode: 'auto' | 'strict' | 'normalize'
): string[] {
  // ... (oryginalna logika)
  // Zostawiamy bez zmian
  return newLines;
}

function validateEdits(edits: Array<{ oldText: string, newText: string }>, debug: boolean, logger: Logger): void {
  // ... (oryginalna logika)
}

function getContextLines(text: string, lineNumber: number, contextSize: number): string {
  // ... (oryginalna logika)
  return '';
}
