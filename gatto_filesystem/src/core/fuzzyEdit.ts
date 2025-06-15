import fs from 'node:fs/promises';
import { createTwoFilesPatch } from 'diff';
import { isBinaryFile } from '../utils/binaryDetect';
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

function doRangesOverlap(range1: AppliedEditRange, range2: {startLine: number, endLine: number}): boolean {
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
    const buffer = await fs.readFile(filePath);
    if (await isBinaryFile(buffer, filePath)) {
      throw createError(
        'BINARY_FILE_ERROR',
        'Cannot edit binary files',
        { filePath, detectedAs: 'binary' }
      );
    }

    const originalContent = normalizeLineEndings(buffer.toString('utf-8'));
    let modifiedContent = originalContent;
    const appliedRanges: AppliedEditRange[] = [];

    validateEdits(edits, config.debug, logger);

    for (const [editIndex, edit] of edits.entries()) {
      let replaced = false;
      const normalizedOld = normalizeLineEndings(edit.oldText);
      const normalizedNew = normalizeLineEndings(edit.newText);

      // caseSensitive: require direct oldText match
      if (config.caseSensitive && !modifiedContent.includes(edit.oldText)) {
        throw createError('NO_MATCH', `Exact text "${edit.oldText}" not found and caseSensitive enabled`);
      }
      // Simple exact match replacement for single-line edits
      if (!normalizedOld.includes('\n') && modifiedContent.includes(config.caseSensitive ? edit.oldText : normalizedOld)) {
        const searchText = config.caseSensitive ? edit.oldText : normalizedOld;
        const replaceText = config.caseSensitive ? edit.newText : normalizedNew;
        modifiedContent = modifiedContent.replace(searchText, replaceText);
        replaced = true;
        continue;
      }

      const exactMatchIndex = modifiedContent.indexOf(normalizedOld);
      if (exactMatchIndex !== -1) {
        replaced = true;
        // Preserve indentation for exact matches using the same logic as fuzzy matches
        const contentLines = modifiedContent.split('\n');
        const oldLinesForIndent = normalizedOld.split('\n');
        const newLinesForIndent = normalizedNew.split('\n');

        // Find the line number of the exact match to get the original indent
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
        const indentedNewLines = applyRelativeIndentation(
          newLinesForIndent,
          oldLinesForIndent,
          originalIndent,
          config.preserveLeadingWhitespace
        );

        // Reconstruct modifiedContent carefully with the new indented lines
        const linesBeforeMatch = modifiedContent.substring(0, exactMatchIndex).split('\n');
        const linesAfterMatch = modifiedContent.substring(exactMatchIndex + normalizedOld.length).split('\n');

        // The new content replaces a certain number of original lines that constituted normalizedOld.
        // We need to splice the contentLines array correctly.
        // The number of lines to replace is oldLinesForIndent.length.
        // The starting line for replacement is lineNumberOfMatch.
        const currentEditTargetRange = {
          startLine: lineNumberOfMatch,
          endLine: lineNumberOfMatch + oldLinesForIndent.length - 1
        };

        for (const appliedRange of appliedRanges) {
          if (doRangesOverlap(appliedRange, currentEditTargetRange)) {
            throw createError(
              'OVERLAPPING_EDIT',
              `Edit ${editIndex + 1} (exact match) overlaps with previously applied edit ${appliedRange.editIndex + 1}. ` +
              `Current edit targets lines ${currentEditTargetRange.startLine + 1}-${currentEditTargetRange.endLine + 1}. ` +
              `Previous edit affected lines ${appliedRange.startLine + 1}-${appliedRange.endLine + 1}.`,
              {
                conflictingEditIndex: editIndex,
                previousEditIndex: appliedRange.editIndex,
                currentEditTargetRange,
                previousEditAffectedRange: appliedRange
              }
            );
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
      } else {
        // Fuzzy match logic
        const contentLines = modifiedContent.split('\n');
        const oldLines = normalizedOld.split('\n');
        const processedOld = preprocessText(normalizedOld, config);

        let bestMatch = {
          distance: Infinity,
          index: -1,
          text: '',
          similarity: 0,
          windowSize: 0
        };
        // ... (tu znajduje się dalsza logika fuzzy match)
        // Jeśli fuzzy match się powiedzie, ustaw replaced = true;
        // Jeśli nie, replaced pozostaje false
      }

      // Po wszystkich próbach: jeśli nie było żadnej podmiany i caseSensitive, rzuć wyjątek
      if (config.caseSensitive && !replaced) {
        throw createError('NO_MATCH', `No match found for edit "${edit.oldText}" (caseSensitive)`);
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

function validateEdits(edits: Array<{oldText: string, newText: string}>, debug: boolean, logger: Logger): void {
  // ... (oryginalna logika)
}

function getContextLines(text: string, lineNumber: number, contextSize: number): string {
  // ... (oryginalna logika)
  return '';
}
