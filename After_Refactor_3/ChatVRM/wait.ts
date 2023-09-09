// utils.ts

/**
 * Delays the execution of a function for a given number of milliseconds.
 * @param ms The number of milliseconds to delay the execution.
 * @returns A Promise that resolves after the specified delay.
 */
export const delay = async (ms: number): Promise<void> =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

// app.ts

import { delay } from './utils';

async function doSomething() {
  console.log('Delaying for 1 second...');
  await delay(1000);
  console.log('Done!');
}

doSomething();

