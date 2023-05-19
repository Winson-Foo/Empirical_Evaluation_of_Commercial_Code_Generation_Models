import { useCallback, useContext, useEffect } from 'react';
import { UNSAFE_NavigationContext as NavigationContext } from 'react-router-dom';

type Tx = {
  retry: () => void;
};

type Blocker = (tx: Tx) => void;

/**
 * A hook that blocks navigation when a certain condition is met.
 * @param blocker A function that determines whether to block navigation.
 * @param when A boolean that determines when to activate the blocker. Default is true.
 */
export function useBlocker(blocker: Blocker, when = true) {
  const { navigator } = useContext(NavigationContext);

  useEffect(() => {
    if (!when) return;

    const unblock = navigator.block((transaction: Tx) => {
      const autoUnblockingTx = {
        ...transaction,
        retry() {
          unblock();
          transaction.retry();
        },
      };

      blocker(autoUnblockingTx);
    });

    return unblock;
  }, [navigator, blocker, when]);
}

/**
 * A hook that prompts the user with a message before blocking navigation.
 * @param message The message to be displayed to the user.
 * @param when A boolean that determines when to activate the prompt. Default is true.
 */
export function usePrompt(message: string, when = true) {
  const promptUser = useCallback(
    (transaction: Tx) => {
      if (window.confirm(message)) transaction.retry();
    },
    [message]
  );

  useBlocker(promptUser, when);
}

