import fs from 'node:fs/promises';
import { createTwoFilesPatch } from 'diff';
import { isBinaryFile } from '../utils/binaryDetect.js';
import { createError } from '../types/errors.js';
import { get as fastLevenshtein } from 'fast-levenshtein';
import { PerformanceTimer } from '../utils/performance.js';
import type { Logger } from 'pino';
import type { EditOperation } from './schemas.js';
import type { Config } from '../server/config.js';

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

// Using fast-levenshtein, the optimized native JS version is no longer primary
// function levenshteinDistanceOptimized(str1: string, str2: string): number {
  if (str1 === str2) return 0;
  if (str1.length === 0) return str2.length;
  if (str2.length === 0) return str1.length;
  
  const shorter = str1.length <= str2.length ? str1 : str2;
  const longer = str1.length <= str2.length ? str2 : str1;
  
  let previousRow = Array(shorter.length + 1).fill(0).map((_, i) => i);
  
  for (let i = 0; i < longer.length; i++) {
    const currentRow = [i + 1];
    for (let j = 0; j < shorter.length; j++) {
      const cost = longer[i] === shorter[j] ? 0 : 1;
      currentRow.push(Math.min(
        currentRow[j] + 1,
        previousRow[j + 1] + 1,
        previousRow[j] + cost
      ));
    }
    previousRow = currentRow;
  }
  // return previousRow[shorter.length];
// }

function calculateSimilarity(distance: number, maxLength: number): number {
  return Math.max(0, 1 - (distance / maxLength));
}

function validateEdits(edits: Array<{oldText: string, newText: string}>, debug: boolean, logger: Logger): void {
  for (let i = 0; i < edits.length; i++) {
    for (let j = i + 1; j < edits.length; j++) {
      const edit1 = edits[i];
      const edit2 = edits[j];
      if (edit1.oldText.includes(edit2.oldText) || edit2.oldText.includes(edit1.oldText)) {
        const warning = `Warning: Potentially overlapping oldText in edits ${i+1} and ${j+1}`;
        logger.warn({ edit1Index: i, edit2Index: j }, warning);
      }
      if (edit1.newText.includes(edit2.oldText) || edit2.newText.includes(edit1.oldText)) {
        const warning = `Warning: newText in edit ${i+1} contains oldText from edit ${j+1} - potential mutual overlap`;
        logger.warn({ edit1Index: i, edit2Index: j }, warning);
      }
    }
  }
}

function applyRelativeIndentation(
  newLines: string[], 
  oldLines: string[], 
  originalIndent: string,
  preserveMode: 'auto' | 'strict' | 'normalize'
): string[] {
  switch (preserveMode) {
    case 'strict':
      return newLines.map(line => originalIndent + line.trimStart());
    case 'normalize':
      return newLines.map(line => originalIndent + line.trimStart());
    case 'auto':
    default:
      return newLines.map((line, idx) => {
        if (idx === 0) {
          return originalIndent + line.trimStart();
        }
        const oldLineIndex = Math.min(idx, oldLines.length - 1);
        const newLineIndent = line.match(/^\s*/)?.[0] || '';
        const baseOldIndent = oldLines[0]?.match(/^\s*/)?.[0]?.length || 0;
        const relativeIndentChange = newLineIndent.length - baseOldIndent;
        const finalIndent = originalIndent + ' '.repeat(Math.max(0, relativeIndentChange));
        return finalIndent + line.trimStart();
      });
  }
}

function getContextLines(text: string, lineNumber: number, contextSize: number): string {
  const lines = text.split('\n');
  const start = Math.max(0, lineNumber - contextSize);
  const end = Math.min(lines.length, lineNumber + contextSize + 1);
  // Return actual line numbers, so add 1 to start index for display if lines are 1-indexed in user's mind
  return lines.slice(start, end).map((line, i) => `${start + i + 1}: ${line}`).join('\n');
}

export async function applyFileEdits(
  filePath: string,
  edits: EditOperation[], // Updated to use EditOperation type
  config: FuzzyMatchConfig,
  logger: Logger,
  globalConfig: Config
): Promise<ApplyFileEditsResult> {
  const timer = new PerformanceTimer('applyFileEdits', logger, globalConfig);
  let levenshteinIterations = 0;
  const appliedRanges: AppliedEditRange[] = [];
  
  try {
    const buffer = await fs.readFile(filePath);
    if (isBinaryFile(buffer, filePath)) {
      throw createError(
        'BINARY_FILE_ERROR',
        'Cannot edit binary files',
        { filePath, detectedAs: 'binary' }
      );
    }

    const originalContent = normalizeLineEndings(buffer.toString('utf-8'));
    let modifiedContent = originalContent;

    validateEdits(edits, config.debug, logger);

    for (const [editIndex, edit] of edits.entries()) {
      const normalizedOld = normalizeLineEndings(edit.oldText);
      const normalizedNew = normalizeLineEndings(edit.newText);
      let matchFound = false;

      const exactMatchIndex = modifiedContent.indexOf(normalizedOld);
      if (exactMatchIndex !== -1) {
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
          charCount += contentLines[i].length + 1; // +1 for newline
        }
        const originalIndent = contentLines[lineNumberOfMatch].match(/^\s*/)?.[0] || '';

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
        matchFound = true;
        appliedRanges.push({
          startLine: currentEditTargetRange.startLine,
          endLine: currentEditTargetRange.startLine + indentedNewLines.length - 1,
          editIndex
        });
      } else {
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

        const windowSizes = [
          oldLines.length,
          Math.max(1, oldLines.length - 1),
          oldLines.length + 1,
          Math.max(1, oldLines.length - 2),
          oldLines.length + 2
        ].filter((size, index, arr) => arr.indexOf(size) === index && size > 0);

        for (const windowSize of windowSizes) {
          if (windowSize > contentLines.length) continue;

          for (let i = 0; i <= contentLines.length - windowSize; i++) {
            const windowLines = contentLines.slice(i, i + windowSize);
            const windowText = windowLines.join('\n');
            const processedWindow = preprocessText(windowText, config);
            
            levenshteinIterations++;
            // Use fast-levenshtein
            const distance = fastLevenshtein(processedOld, processedWindow);
            const similarity = calculateSimilarity(distance, Math.max(processedOld.length, processedWindow.length));

            if (distance < bestMatch.distance) {
              bestMatch = { distance, index: i, text: windowText, similarity, windowSize };
            }
            if (distance === 0) break;
          }
          if (bestMatch.distance === 0) break;
        }

        const distanceThreshold = Math.floor(processedOld.length * config.maxDistanceRatio);

        if (bestMatch.index !== -1 && 
            bestMatch.distance <= distanceThreshold && 
            bestMatch.similarity >= config.minSimilarity) {
          
          const newLines = normalizedNew.split('\n');
          const originalIndent = contentLines[bestMatch.index].match(/^\s*/)?.[0] || '';
          const indentedNewLines = applyRelativeIndentation(
            newLines, 
            oldLines, 
            originalIndent, 
            config.preserveLeadingWhitespace
          );

          contentLines.splice(bestMatch.index, bestMatch.windowSize, ...indentedNewLines);
          modifiedContent = contentLines.join('\n');
          matchFound = true;
        } else if (bestMatch.similarity >= 0.5) {
          if (edit.forcePartialMatch) {
            logger.warn(`Applying forced partial match for edit ${editIndex + 1} (similarity: ${bestMatch.similarity.toFixed(3)}) due to 'forcePartialMatch: true'.`);
            const newLines = normalizedNew.split('\n');
            const originalIndent = contentLines[bestMatch.index].match(/^\s*/)?.[0] || '';
            const oldLinesForIndent = normalizedOld.split('\n');
            const indentedNewLines = applyRelativeIndentation(
              newLines, 
              oldLinesForIndent,
              originalIndent, 
              config.preserveLeadingWhitespace
            );
            const currentEditTargetRange = {
              startLine: bestMatch.index,
              endLine: bestMatch.index + bestMatch.windowSize - 1
            };

            for (const appliedRange of appliedRanges) {
              if (doRangesOverlap(appliedRange, currentEditTargetRange)) {
                throw createError(
                  'OVERLAPPING_EDIT',
                  `Edit ${editIndex + 1} (fuzzy match at line ${bestMatch.index + 1}) overlaps with previously applied edit ${appliedRange.editIndex + 1}. ` +
                  `Current edit targets lines ${currentEditTargetRange.startLine + 1}-${currentEditTargetRange.endLine + 1}. ` +
                  `Previous edit affected lines ${appliedRange.startLine + 1}-${appliedRange.endLine + 1}.`,
                  {
                    conflictingEditIndex: editIndex,
                    previousEditIndex: appliedRange.editIndex,
                    currentEditTargetRange,
                    previousEditAffectedRange: appliedRange,
                    similarity: bestMatch.similarity
                  }
                );
              }
            }

            contentLines.splice(bestMatch.index, bestMatch.windowSize, ...indentedNewLines);
            modifiedContent = contentLines.join('\n');
            matchFound = true;
            appliedRanges.push({
              startLine: currentEditTargetRange.startLine,
              endLine: currentEditTargetRange.startLine + indentedNewLines.length - 1,
              editIndex
            });
          } else {
            const contextText = getContextLines(normalizeLineEndings(originalContent), bestMatch.index, 3);
            const partialDiff = createUnifiedDiff(bestMatch.text, processedOld, filePath);
            throw createError(
              'PARTIAL_MATCH',
              `Partial match found for edit ${editIndex + 1} (similarity: ${bestMatch.similarity.toFixed(3)}).` +
              `\n=== Context (around line ${bestMatch.index + 1} in preprocessed content) ===\n${contextText}\n` +
              `\n=== Diff (actual found text vs. your preprocessed oldText) ===\n${partialDiff}\n` +
              `\n=== Suggested Fix ===\n` +
              `1. Adjust 'oldText' to match the content more closely.\n` +
              `2. Or set 'forcePartialMatch: true' for this edit operation if this partial match is acceptable.`, 
              {
                editIndex,
                similarity: bestMatch.similarity,
                bestMatchPreview: bestMatch.text.substring(0, 100),
                context: contextText,
                diff: partialDiff
              }
            );
          }
        }
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