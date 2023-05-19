import { useEffect } from 'react';
import useApi from './useApi';
import useStore, { setShareToken } from 'store/app';

// Selector to get the share token from the store
const selectShareToken = state => state.shareToken;

// Custom hook to fetch and return the share token
export default function useShareToken(shareId) {
  // Get the share token from the store
  const shareToken = useStore(selectShareToken);

  // Get the API request function from the useApi hook
  const { get } = useApi();

  // Function to fetch the share token from the API
  async function fetchShareToken(id) {
    try {
      const data = await get(`/share/${id}`);
      if (data) {
        setShareToken(data);
      }
    } catch (error) {
      console.error(error);
      // Handle error here
    }
  }

  // Use the useEffect hook to fetch the share token when shareId changes
  useEffect(() => {
    if (shareId) {
      fetchShareToken(shareId);
    }
  }, [shareId]);

  // Return the share token from the store
  return shareToken;
}

