import { useEffect } from 'react';
import { Loading } from 'react-basics';
import { useRouter } from 'next/router';
import { setClientAuthToken } from 'lib/client';

function RedirectToUrl({ url }) {
  const router = useRouter();

  useEffect(() => {
    router.push(url);
  }, [router, url]);

  return <Loading size="xl" />;
}

export default function SingleSignOnPage() {
  const { token, url } = useRouter().query;

  useEffect(() => {
    if (url && token) {
      setClientAuthToken(token);
    }
  }, [token, url]);

  return url && token ? <RedirectToUrl url={url} /> : <Loading size="xl" />;
}