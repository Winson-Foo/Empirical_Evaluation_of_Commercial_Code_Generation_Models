import { useCallback, useState } from 'react';

/**
 * Hook that forces a re-render of a component
 * @return {Function} A callback that triggers a re-render
 */
export default function useForceUpdate() {
  const [, setObject] = useState(Object.create(null));

  // Return a memoized callback that updates an object in state
  // Every time the callback is called, the object changes, triggering a re-render
  return useCallback(() => {
    setObject(Object.create(null));
  }, [setObject]);
}

