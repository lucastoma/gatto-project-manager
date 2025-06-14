export interface HintInfo {
  confidence: number;
  hint: string;
  example?: unknown;
}

export const HINTS: Record<string, HintInfo> = {
  ACCESS_DENIED: {
    confidence: 0.9,
    hint: "Path is outside of allowedDirectories. Check the 'path' argument.",
    example: { path: "/workspace/project/file.txt" }
  },
  VALIDATION_ERROR: { confidence: 0.8, hint: "Check the required fields and their types in the arguments." },
  FUZZY_MATCH_FAILED: {
    confidence: 0.7,
    hint: "Try adjusting minSimilarity/maxDistanceRatio or check for whitespace/indentation issues.",
    example: { minSimilarity: 0.6, maxDistanceRatio: 0.3 }
  },
  BINARY_FILE_ERROR: {
    confidence: 0.95,
    hint: "'edit_file' works only on text files. Use 'write_file' with base64 encoding for binary files."
  },
  PARTIAL_MATCH: {
    confidence: 0.6,
    hint: "Found a partial match. Review the diff and correct 'oldText' or parameters."
  },
  FILE_NOT_FOUND_MULTI: {
    confidence: 0.8,
    hint: "One or more requested files could not be read. See per-file result details."
  },
  UNKNOWN_TOOL: {
    confidence: 0.9,
    hint: "The requested tool does not exist. Use 'list_tools' to see available tools."
  },
  DEST_EXISTS: {
    confidence: 0.85,
    hint: "Destination path already exists. Provide a different destination or remove the existing file first.",
    example: { source: "/workspace/src.txt", destination: "/workspace/dest.txt" }
  },
  SRC_MISSING: {
    confidence: 0.85,
    hint: "Source file does not exist. Check the 'source' argument.",
    example: { source: "/workspace/missing.txt" }
  }
  UNKNOWN_ERROR: {
    confidence: 0.1,
    hint: "An unexpected error occurred. Check server logs for details."
  }
};
