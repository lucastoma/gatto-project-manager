import { HintInfo, HINTS } from "../utils/hintMap.js";

export interface StructuredError {
  code: keyof typeof HINTS | string;
  message: string;
  hint?: HintInfo["hint"];
  confidence?: HintInfo["confidence"];
  details?: unknown;
}

export function createError(
  code: StructuredError["code"],
  message: string,
  details?: unknown
): StructuredError {
  const hint = HINTS[code as keyof typeof HINTS];
  return {
    code,
    message,
    hint: hint?.hint,
    confidence: hint?.confidence,
    details
  };
}
