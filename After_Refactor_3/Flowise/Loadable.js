import React, { Suspense } from 'react';
import PropTypes from 'prop-types';

import Loader from './Loader';

const LazyLoad = (Component) => (props) => (
  <Suspense fallback={<Loader />}>
    <Component {...props} />
  </Suspense>
);

LazyLoad.propTypes = {
  component: PropTypes.oneOfType([
    PropTypes.func.isRequired,
    PropTypes.object.isRequired,
  ]),
};

export default LazyLoad;

