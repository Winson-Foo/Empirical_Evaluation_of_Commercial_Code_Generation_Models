import { useCallback, useContext, useEffect } from 'react';
import { UNSAFE_NavigationContext as NavigationContext } from 'react-router-dom';

/**
 * Hook to display a prompt on navigation
 * @param {string} message - The message to display in the prompt
 * @param {boolean} when - Whether to display the prompt or not
 */
export function usePrompt(message: string, when = true): void {
  /**
   * Function to block navigation and display a prompt
   * @param {Object} tx - The transaction object provided by navigator.block()
   */
  const promptBlocker = useCallback(
    (tx: { retry: () => void }): void => {
      if (window.confirm(message)) tx.retry();
    },
    [message]
  );

  useBlocker(promptBlocker, when);
}

/**
 * Hook to block navigation
 * @param {function} blocker - The function to handle blocking of navigation
 * @param {boolean} when - Whether to block navigation or not
 */
export function useBlocker(blocker: (tx: { retry: () => void }) => void, when = true): void {
  const { navigator } = useContext(NavigationContext);

  useEffect(() => {
    if (!when) return;

    const unblock = navigator.block((tx) => {
      const autoUnblockingTx = {
        ...tx,
        retry() {
          unblock();
          tx.retry();
        },
      };

      blocker(autoUnblockingTx);
    });

    return unblock;
  }, [navigator, blocker, when]);
} 