import { useEffect } from 'react';
import { Loading } from 'react-basics';
import { useRouter } from 'next/router';
import { setClientAuthToken } from 'lib/client';

const CLIENT_AUTH_TOKEN = 'clientAuthToken';

export default function SingleSignOnPage() {
  const router = useRouter();
  const { token, url } = router.query;

  useEffect(() => {
    // Extract logic to reusable function that sets client auth token
    if (url && token) {
      setAuthToken(token);

      redirectToPage(url);
    }
  }, [router, url, token]);

  return <Loading size="xl" />;
}

// Helper function to set client auth token
const setAuthToken = (token) => {
  localStorage.setItem(CLIENT_AUTH_TOKEN, token);
};

// Helper function to redirect to a given URL
const redirectToPage = (url) => {
  router.push(url);
};

