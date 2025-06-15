import fs from 'node:fs/promises';
import { createTwoFilesPatch } from 'diff';
import { isBinaryFile } from '../utils/binaryDetect.js';
import { createError } from '../types/errors.js';
import { PerformanceTimer } from '../utils/performance.js';
import type { Logger } from 'pino';
import type { EditOperation } from './schemas.js';
import type { Config } from '../server/config.js';

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

function levenshteinDistanceOptimized(str1: string, str2: string): number {
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
  return previousRow[shorter.length];
}

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
        modifiedContent = modifiedContent.substring(0, exactMatchIndex) +
                          normalizedNew +
                          modifiedContent.substring(exactMatchIndex + normalizedOld.length);
        matchFound = true;
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
            const distance = levenshteinDistanceOptimized(processedOld, processedWindow);
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
            contentLines.splice(bestMatch.index, bestMatch.windowSize, ...indentedNewLines);
            modifiedContent = contentLines.join('\n');
            matchFound = true;
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
    const formattedDiff = "```diff\n" + diff + "\n```\n\n";

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