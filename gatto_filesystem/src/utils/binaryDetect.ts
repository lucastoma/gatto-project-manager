import path from 'node:path';
import { isUtf8 as bufferIsUtf8 } from 'buffer';
import { lookup } from 'mime-types';

const NUL_BYTE_CHECK_SAMPLE_SIZE = 8192; // Check first 8KB for NUL bytes

// Enum for file type classification
export enum FileType {
  TEXT,
  POTENTIAL_TEXT_WITH_CAVEATS, // e.g., text with NULs, or very large but seems text-based
  CONFIRMED_BINARY
}

// More conservative list of extensions that are almost certainly binary
const STRICT_BINARY_EXTENSIONS = new Set([
  '.exe', '.dll', '.so', '.dylib', '.bin', // .dat removed, see note below
  '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.ico',
  '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
  '.pdf', '.zip', '.rar', '.tar', '.gz', '.7z',
  '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
  '.wasm', '.o', '.a', '.obj', '.lib', '.class', '.pyc', '.pyo', '.pyd', // Compiled code/objects
  '.sqlite', '.db', '.mdb', '.accdb', '.swf', '.fla', // Databases and flash
  // Note: .dat is removed as it's too ambiguous. Content checks will be more important.
]);

// List of MIME types that strongly indicate a binary file
const BINARY_MIME_TYPE_PATTERNS = [
  /^image\//,
  /^audio\//,
  /^video\//,
  /^application\/(zip|x-rar-compressed|x-tar|gzip|x-7z-compressed)$/,
  /^application\/(pdf|vnd\.openxmlformats-officedocument.*|msword|octet-stream)$/
];

export function classifyFileType(buffer: Buffer, filename?: string): FileType {
  const isUtf8 = (Buffer as any).isUtf8 ?? bufferIsUtf8;
  const isUtf8Internal = (Buffer as any).isUtf8 ?? bufferIsUtf8;

  // Check 1: Strict extension check
  if (filename) {
    const ext = path.extname(filename).toLowerCase();
    if (STRICT_BINARY_EXTENSIONS.has(ext)) {
      return FileType.CONFIRMED_BINARY;
    }

    // Check 2: MIME type check
    const mimeType = lookup(filename);
    if (mimeType) {
      for (const pattern of BINARY_MIME_TYPE_PATTERNS) {
        if (pattern.test(mimeType)) {
          return FileType.CONFIRMED_BINARY;
        }
      }
    }
  }

  // Check 3: UTF-8 validity
  // If a buffer is not valid UTF-8, it's highly likely binary or an unsupported encoding.
  if (!isUtf8Internal(buffer)) {
    return FileType.CONFIRMED_BINARY;
  }

  // Check 4: Non-printable character percentage (strong indicator for binary)
  let nonPrintable = 0;
  const checkLength = Math.min(buffer.length, 1024); // Check up to the first 1KB
  for (let i = 0; i < checkLength; i++) {
    const byte = buffer[i];
    // Allow TAB, LF, CR. Consider other control characters as non-printable for this check.
    if (byte < 32 && byte !== 9 && byte !== 10 && byte !== 13) {
      nonPrintable++;
    }
  }
  // If a significant portion of the initial part is non-printable, classify as binary.
  // Threshold can be adjusted. 0.1 means 10%.
  if (checkLength > 0 && (nonPrintable / checkLength) > 0.10) {
    return FileType.CONFIRMED_BINARY;
  }

  // Check 5: NUL bytes (indicator for POTENTIAL_TEXT_WITH_CAVEATS if not already CONFIRMED_BINARY)
  // Check for NUL bytes only in a sample of the buffer to avoid performance issues with large files.
  const sampleForNulCheck = buffer.length > NUL_BYTE_CHECK_SAMPLE_SIZE 
    ? buffer.subarray(0, NUL_BYTE_CHECK_SAMPLE_SIZE) 
    : buffer;
  if (sampleForNulCheck.includes(0)) {
    return FileType.POTENTIAL_TEXT_WITH_CAVEATS;
  }

  // If none of the above, assume it's text
  return FileType.TEXT;
}
