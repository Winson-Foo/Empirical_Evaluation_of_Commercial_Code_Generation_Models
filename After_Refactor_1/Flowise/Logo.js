// flowiseLogo.js

import React from 'react';
import PropTypes from 'prop-types';
import { useSelector } from 'react-redux';
import styles from 'assets/css/flowiseLogo.module.css';

const LOGO_URL = 'assets/images/flowise_logo.png';
const LOGO_DARK_URL = 'assets/images/flowise_logo_dark.png';

const FlowiseLogo = ({ altText }) => {
  const { isDarkMode } = useSelector((state) => state.customization);

  return (
    <div className={styles.logoContainer}>
      <img
        className={styles.logoImage}
        src={isDarkMode ? LOGO_DARK_URL : LOGO_URL}
        alt={altText}
      />
    </div>
  );
};

FlowiseLogo.propTypes = {
  altText: PropTypes.string.isRequired,
};

export default FlowiseLogo;

// flowiseLogo.module.css

.logoContainer {
  align-items: center;
  display: flex;
  flex-direction: row;
}

.logoImage {
  object-fit: contain;
  height: auto;
  width: 150px;
}

