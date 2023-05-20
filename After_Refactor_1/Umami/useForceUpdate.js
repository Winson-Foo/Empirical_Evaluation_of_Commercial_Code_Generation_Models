import { useCallback, useState } from 'react';

export default function useForceUpdate() {
  const [_, setUpdate] = useState(Object.create(null));

  // this function is called to force an update of a component
  const forceUpdate = useCallback(() => {
    setUpdate(Object.create(null));
  }, [setUpdate]);

  return forceUpdate;
}

