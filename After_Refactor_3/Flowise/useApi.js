import { useState } from 'react'

type RequestType<T> = {
  data: T | null,
  error: string | null,
  loading: boolean,
  request: (...args: any[]) => Promise<void>
}

export default function useRequest<T> (apiFunc: (...args: any[]) => Promise<{ data: T }>, errorMessage: string = 'Unexpected error'): RequestType<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState<boolean>(false)

  const request = async (...args: any[]) => {
    setLoading(true)
    try {
      const result = await apiFunc(...args)
      setData(result.data)
    } catch (err) {
      setError(err.message || errorMessage)
    } finally {
      setLoading(false)
    }
  }

  return {
    data,
    error,
    loading,
    request
  }
}

