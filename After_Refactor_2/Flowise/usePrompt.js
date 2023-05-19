import { useCallback, useContext, useEffect } from 'react'
import { NavigationContext } from 'react-router-dom'

type Tx = {
  retry: () => void
}

type Blocker = (tx: Tx) => void

export function useBlocker(blocker: Blocker, when: boolean = true): void {
  const { navigator } = useContext(NavigationContext)

  useEffect(() => {
    if (!when) return

    const unblock = navigator.block((tx: Tx) => {
      const autoUnblockingTx = {
        ...tx,
        retry() {
          unblock()
          tx.retry()
        },
      }

      blocker(autoUnblockingTx)
    })

    return unblock
  }, [navigator, blocker, when])
}

export function usePrompt(message: string, when: boolean = true): void {
  const handleBlocker = useCallback(
    (tx: Tx) => {
      if (window.confirm(message)) tx.retry()
    },
    [message]
  )

  useBlocker(handleBlocker, when)
}

