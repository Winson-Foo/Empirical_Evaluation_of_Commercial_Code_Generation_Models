/**
 * Delays the execution of code for a given number of milliseconds
 * @param delayMs The time to delay, in milliseconds
 */
export const delay = async (delayMs: number): Promise<any> => {
  try {
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  } catch (error) {
    console.error(`An error occurred while delaying execution: ${error}`);
  }
};