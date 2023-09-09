import { useCallback, useState } from 'react';

type Callback = () => void;

export default function useForceUpdate(): Callback {
  const [unusedState, setState] = useState<{}>(Object.create(null));

  // Update the state with a new object to trigger a re-render
  const forceUpdate: Callback = useCallback(() => {
    setState(Object.create(null));
  }, [setState]);

  return forceUpdate;
} 