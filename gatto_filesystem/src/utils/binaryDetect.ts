import path from 'node:path';
import { isUtf8 as bufferIsUtf8 } from 'buffer';

const BINARY_EXTENSIONS = new Set([
  '.exe', '.dll', '.so', '.dylib', '.bin', '.dat',
  '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.ico',
  '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
  '.pdf', '.zip', '.rar', '.tar', '.gz', '.7z',
  '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'
]);

export function isBinaryFile(buffer: Buffer, filename?: string): boolean {
  const isUtf8 = (Buffer as any).isUtf8 ?? bufferIsUtf8;
  if (isUtf8 && !isUtf8(buffer)) {
    return true;
  }

  if (buffer.includes(0)) {
    return true;
  }

  if (filename) {
    const ext = path.extname(filename).toLowerCase();
    if (BINARY_EXTENSIONS.has(ext)) {
      return true;
    }
  }

  let nonPrintable = 0;
  const sampleSize = Math.min(1024, buffer.length);
  
  for (let i = 0; i < sampleSize; i++) {
    const byte = buffer[i];
    if (byte < 32 && byte !== 9 && byte !== 10 && byte !== 13) {
      nonPrintable++;
    }
  }

  return (nonPrintable / sampleSize) > 0.1;
}
