import React, { Suspense } from 'react';
import Loader from './Loader';

// HOC to add lazy loading and a loader for suspense fallback
const withLoadable = (WrappedComponent) => {
  const LoadableComponent = (props) => {
    const { ...rest } = props;

    return (
      <Suspense fallback={<Loader />}>
        <WrappedComponent {...rest} />
      </Suspense>
    );
  };

  return LoadableComponent;
};

export default withLoadable;

