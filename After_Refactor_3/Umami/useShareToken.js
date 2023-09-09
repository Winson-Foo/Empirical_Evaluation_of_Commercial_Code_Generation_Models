import { useEffect } from 'react';

import useApi from './useApi';
import useStore, { setShareToken } from 'store/app';

/**
 * Returns the share token from the store.
 */
const selector = state => state.shareToken;

/**
 * Loads the share token from the API and updates the store.
 * @param {string} id - The share ID to load the token for.
 */
async function fetchShareToken(id) {
  const data = await get(`/share/${id}`);
  if (data) {
    setShareToken(data);
  }
}

/**
 * Custom hook that fetches the share token from the API.
 * @param {string} shareId - The ID of the share to fetch the token for.
 * @returns {string} - The share token.
 */
export default function useFetchShareToken(shareId) {
  const shareToken = useStore(selector);
  const { get } = useApi();

  useEffect(() => {
    if (shareId) {
      fetchShareToken(shareId);
    }
  }, [shareId]);

  return shareToken;
}

