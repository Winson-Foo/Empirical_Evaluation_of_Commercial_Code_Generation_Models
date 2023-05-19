import { useState } from 'react'

export default (apiFunction) => {
  const [responseData, setResponseData] = useState(null)
  const [errorMessage, setErrorMessage] = useState(null)
  const [isLoading, setLoading] = useState(false)

  const makeApiCall = async (...args) => {
    try {
      setLoading(true)
      const response = await apiFunction(...args)
      setResponseData(response.data || null)
    } catch (error) {
      setErrorMessage(
        `There was an error fetching data from the server: ${error.message}`
      )
      setResponseData(null)
    } finally {
      setLoading(false)
    }
  }

  return {
    responseData,
    errorMessage,
    isLoading,
    makeApiCall,
  }
}

