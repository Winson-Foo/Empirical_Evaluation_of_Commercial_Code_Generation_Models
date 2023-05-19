import { useEffect } from 'react';
import { Loading } from 'react-basics';
import { useRouter } from 'next/router';
import { setClientAuthToken } from 'lib/client';

// This function handles the Single Sign-On logic
function handleSSO(token, url, router) {
  if (url && token) {
    setClientAuthToken(token);
    router.push(url); // Redirect to the specified URL
  }
}

export default function SingleSignOnPage() {
  const router = useRouter();
  const { token, url } = router.query;

  useEffect(() => {
    handleSSO(token, url, router);
  }, [router, url, token]);

  return <Loading size="xl" />;
} 