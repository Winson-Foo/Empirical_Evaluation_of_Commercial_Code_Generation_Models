import { useState } from 'react';

const useApi = (apiFunction) => {
  const [responseData, setResponseData] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const makeRequest = async (...args) => {
    setIsLoading(true);
    try {
      const response = await apiFunction(...args);
      setResponseData(response.data);
    } catch (error) {
      setErrorMessage(error.message || 'Unexpected error');
    } finally {
      setIsLoading(false);
    }
  };

  return {
    responseData,
    errorMessage,
    isLoading,
    makeRequest,
  };
};

export default useApi;

