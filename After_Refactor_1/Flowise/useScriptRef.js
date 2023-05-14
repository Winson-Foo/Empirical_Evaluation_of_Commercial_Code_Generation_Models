// This hook returns a reference to a boolean value that can be used to check if a component is still mounted
import { useEffect, useRef } from 'react'

const useMountedRef = () => {
  const mountedRef = useRef(true)

  useEffect(() => {
    // Set mountedRef to false when the component unmounts
    return () => {
      mountedRef.current = false
    }
  }, [])

  return mountedRef
}

export default useMountedRef

