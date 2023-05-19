export const delay = async (durationMs: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, durationMs));