import { performance } from 'node:perf_hooks';
import type { Logger } from 'pino';
import type { Config } from '../server/config.js';

export class PerformanceTimer {
  private startTime: number;
  private operation: string;
  private logger: Logger;
  private enabled: boolean;

  constructor(operation: string, logger: Logger, config: Config) {
    this.operation = operation;
    this.logger = logger;
    this.enabled = config.logging.performance;
    this.startTime = this.enabled ? performance.now() : 0;
  }

  end(additionalData?: any): number {
    if (!this.enabled) {
      return 0;
    }
    const duration = performance.now() - this.startTime;
    this.logger.debug({
      operation: this.operation,
      duration_ms: Math.round(duration * 100) / 100,
      ...additionalData
    }, `Performance: ${this.operation}`);
    return duration;
  }
}
