/// <reference types="jest" />

import { classifyFileType, FileType } from '../binaryDetect';
import { Buffer } from 'buffer';

// Mock mime-types lookup
// We'll need to enhance this mock for more specific MIME type tests
jest.mock('mime-types', () => ({
  lookup: jest.fn((filename: string) => {
    if (filename.endsWith('.jpg')) return 'image/jpeg';
    if (filename.endsWith('.png')) return 'image/png';
    if (filename.endsWith('.pdf')) return 'application/pdf';
    if (filename.endsWith('.zip')) return 'application/zip';
    if (filename.endsWith('.txt')) return 'text/plain';
    if (filename.endsWith('.csv')) return 'text/csv';
    if (filename.endsWith('.log')) return 'text/plain'; // Or application/octet-stream if unknown by mime
    if (filename.endsWith('.dat')) return false; // Simulate unknown MIME for .dat
    return false; // Default for unknown types
  }),
}));

describe('classifyFileType', () => {
  // Test cases for CONFIRMED_BINARY
  describe('CONFIRMED_BINARY', () => {
    it('should classify files with STRICT_BINARY_EXTENSIONS as CONFIRMED_BINARY', () => {
      const buffer = Buffer.from('some content');
      expect(classifyFileType(buffer, 'test.exe')).toBe(FileType.CONFIRMED_BINARY);
      expect(classifyFileType(buffer, 'image.jpg')).toBe(FileType.CONFIRMED_BINARY);
      expect(classifyFileType(buffer, 'archive.zip')).toBe(FileType.CONFIRMED_BINARY);
      expect(classifyFileType(buffer, 'document.pdf')).toBe(FileType.CONFIRMED_BINARY);
      expect(classifyFileType(buffer, 'compiled.wasm')).toBe(FileType.CONFIRMED_BINARY);
    });

    it('should classify files with binary MIME types as CONFIRMED_BINARY', () => {
      const buffer = Buffer.from('some content');
      // Mocked 'mime-types' will return 'image/jpeg' for .jpg
      expect(classifyFileType(buffer, 'photo.jpg')).toBe(FileType.CONFIRMED_BINARY);
      // Mocked 'mime-types' will return 'application/pdf' for .pdf
      expect(classifyFileType(buffer, 'manual.pdf')).toBe(FileType.CONFIRMED_BINARY);
    });

    it('should classify non-UTF8 buffers as CONFIRMED_BINARY', () => {
      // Create a buffer with invalid UTF-8 sequence (0xFF is not valid in UTF-8)
      const nonUtf8Buffer = Buffer.from([0x48, 0x65, 0x6C, 0x6C, 0x6F, 0xFF]); // "Hello" + invalid byte
      expect(classifyFileType(nonUtf8Buffer, 'test.txt')).toBe(FileType.CONFIRMED_BINARY);
    });

    it('should classify files with high non-printable char ratio as CONFIRMED_BINARY', () => {
      // Buffer with >10% non-printable chars (e.g., many control chars < 32, excluding tab, lf, cr)
      const nonPrintableByteArray = Array(100).fill(0x01); // 100 SOH characters as byte values
      const mostlyNonPrintableBuffer = Buffer.from(nonPrintableByteArray);
      expect(classifyFileType(mostlyNonPrintableBuffer, 'control.dat')).toBe(FileType.CONFIRMED_BINARY);
    });
  });

  // Test cases for POTENTIAL_TEXT_WITH_CAVEATS
  describe('POTENTIAL_TEXT_WITH_CAVEATS', () => {
    it('should classify UTF-8 text files with NUL bytes as POTENTIAL_TEXT_WITH_CAVEATS', () => {
      const textWithNul = Buffer.from('This is a text file with a \x00 NUL byte.');
      expect(classifyFileType(textWithNul, 'text_with_nul.txt')).toBe(FileType.POTENTIAL_TEXT_WITH_CAVEATS);
    });

    it('should classify CSV with NUL bytes as POTENTIAL_TEXT_WITH_CAVEATS', () => {
      const csvWithNul = Buffer.from('col1,col2\nval1,val\x002'); // NUL byte followed by '2'
      expect(classifyFileType(csvWithNul, 'data.csv')).toBe(FileType.POTENTIAL_TEXT_WITH_CAVEATS);
    });
  });

  // Test cases for TEXT
  describe('TEXT', () => {
    it('should classify clean UTF-8 text files as TEXT', () => {
      const cleanText = Buffer.from('This is a perfectly normal text file.');
      expect(classifyFileType(cleanText, 'clean.txt')).toBe(FileType.TEXT);
    });

    it('should classify clean CSV files as TEXT', () => {
      const cleanCsv = Buffer.from('header1,header2\nvalue1,value2');
      expect(classifyFileType(cleanCsv, 'data.csv')).toBe(FileType.TEXT);
    });

    it('should classify clean log files as TEXT', () => {
      const cleanLog = Buffer.from('[INFO] This is a log entry.');
      expect(classifyFileType(cleanLog, 'app.log')).toBe(FileType.TEXT);
    });

    it('should classify a .dat file with text content as TEXT (if no NULs and UTF8)', () => {
      const textDat = Buffer.from('Some textual data in a .dat file');
      // .dat is not in STRICT_BINARY_EXTENSIONS, and mime-types mock returns false for .dat
      expect(classifyFileType(textDat, 'config.dat')).toBe(FileType.TEXT);
    });
  });

  // Edge cases and ambiguous files
  describe('Edge Cases', () => {
    it('should handle empty files as TEXT', () => {
      const emptyBuffer = Buffer.from('');
      expect(classifyFileType(emptyBuffer, 'empty.txt')).toBe(FileType.TEXT);
    });

    it('should handle a .dat file with NUL bytes as POTENTIAL_TEXT_WITH_CAVEATS', () => {
      const datWithNul = Buffer.from('data with\x00nul');
      expect(classifyFileType(datWithNul, 'ambiguous.dat')).toBe(FileType.POTENTIAL_TEXT_WITH_CAVEATS);
    });

    it('should handle a .dat file with binary-like content (non-UTF8) as CONFIRMED_BINARY', () => {
      const binaryDat = Buffer.from([0x01, 0x02, 0x03, 0xFF]);
      expect(classifyFileType(binaryDat, 'binary.dat')).toBe(FileType.CONFIRMED_BINARY);
    });
  });
});
