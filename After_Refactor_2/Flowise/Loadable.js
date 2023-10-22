import { lazy, Suspense } from 'react';
import Loader from './Loader';

// Use lazy loading and cache the component for future use
const Loadable = (importFunc) => {
  const Component = lazy(importFunc);
  return (props) => (
    <Suspense fallback={<Loader />}>
      <Component {...props} />
    </Suspense>
  );
};

export default Loadable;