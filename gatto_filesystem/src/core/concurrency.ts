import { Semaphore } from 'async-mutex';
import type { Config } from '../server/config';

let globalEditSemaphore: Semaphore;

export function initGlobalSemaphore(config: Config) {
  globalEditSemaphore = new Semaphore(config.concurrency.maxGlobalConcurrentEdits ?? 20);
}

export function getGlobalSemaphore() {
  if (!globalEditSemaphore) {
    throw new Error('Global semaphore not initialized');
  }
  return globalEditSemaphore;
}