import { useEffect } from 'react';
import useApi from './useApi';
import useStore, { setShareToken } from 'store/app';

const selector = (state) => state.shareToken;

export default function useShareToken(shareId) {
  const shareToken = useStore(selector);
  const { get } = useApi();

  async function loadToken(id) {
    try {
      const data = await get(`/share/${id}`);
      setShareToken(data);
    } catch (error) {
      console.error(error);
    }
  }

  useEffect(() => {
    if (shareId) {
      loadToken(shareId);
    }
  }, [shareId]);

  return shareToken;
}